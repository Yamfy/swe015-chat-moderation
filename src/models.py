"""Common interface for the 4-model ladder (+ the rule baseline).

Every model exposes:
    name, precision, files (for size), predict_proba(list[str]) -> np.ndarray of P(offensive)
"""
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

from . import config
from .lexicon import LexiconMatcher
from .normalize import normalize, normalize_for_bert
from .w2v import FFN, average_embeddings, load_w2v

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
A = config.ART_DIR

PATHS = {
    "lexicon": [A / "lexicon.json"],
    "tfidf": [A / "tfidf_lr.joblib"],
    "w2v_ffn": [A / "w2v_vocab.json", A / "w2v_emb.npy", A / "ffn.pt"],
    "bert_fp32": [A / "bert_fp32"],
    "bert_int8": [A / "bert_int8"],
}


def size_mb(paths):
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # weights only (tokenizer / config files are shared by fp32 and int8)
            total += sum(f.stat().st_size for f in p.iterdir()
                         if f.suffix in (".safetensors", ".bin", ".pt"))
        elif p.exists():
            total += p.stat().st_size
    return total / 2 ** 20


class Lexicon:
    key, name, precision = "lexicon", "Rules: lexicon + edit distance", "n/a"

    def __init__(self):
        self.m = LexiconMatcher.load(PATHS["lexicon"][0])

    def predict_proba(self, texts):
        return np.array([self.m.predict_proba(t) for t in texts])


class TfidfLR:
    key, name, precision = "tfidf", "Baseline: tf-idf + logistic regression", "FP64"

    def __init__(self):
        self.pipe = joblib.load(PATHS["tfidf"][0])

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1]


class W2VFFN:
    key, name, precision = "w2v_ffn", "NN: own word2vec + feedforward net", "FP32"

    def __init__(self):
        v, e, f = PATHS["w2v_ffn"]
        self.stoi, self.emb = load_w2v(v, e)
        self.net = FFN(dim=self.emb.shape[1])
        self.net.load_state_dict(torch.load(f, map_location="cpu"))
        self.net.eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        x = torch.from_numpy(average_embeddings(texts, self.stoi, self.emb))
        return torch.softmax(self.net(x), -1)[:, 1].numpy()


class Bert:
    """BERTurk fine-tuned for clean/offensive. precision = 'FP32' or 'INT8' (dynamic quantization)."""

    def __init__(self, int8: bool, device="cpu"):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.key = "bert_int8" if int8 else "bert_fp32"
        self.precision = "INT8 (dynamic)" if int8 else "FP32"
        self.name = f"Transformer: BERTurk fine-tuned ({'INT8' if int8 else 'FP32'})"
        d = PATHS[self.key][0]
        self.tok = AutoTokenizer.from_pretrained(d)
        if int8:
            torch.backends.quantized.engine = _quant_engine()
            cfg = AutoConfig.from_pretrained(d)
            try:
                model = quantize(AutoModelForSequenceClassification.from_config(cfg))
                # our own file; quantized packed params need weights_only=False to unpickle
                model.load_state_dict(torch.load(d / "model_int8.pt", map_location="cpu",
                                                 weights_only=False))
            except Exception as e:  # e.g. file packed for x86 loaded on Apple Silicon (qnnpack)
                print(f"[INT8] re-quantizing FP32 weights on this CPU ({type(e).__name__})")
                model = quantize(AutoModelForSequenceClassification.from_pretrained(PATHS["bert_fp32"][0]))
            device = "cpu"                     # dynamic INT8 kernels are CPU-only
        else:
            model = AutoModelForSequenceClassification.from_pretrained(d)
        self.device = device
        self.model = model.to(device).eval()

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok([normalize_for_bert(t) for t in texts[i:i + batch_size]],
                           truncation=True, max_length=config.MAX_LEN, padding=True,
                           return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out)


def _quant_engine():
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):      # qnnpack = Apple Silicon / ARM
        if e in engines:
            return e
    return engines[0]


def quantize(model):
    """Post-training dynamic quantization: every nn.Linear -> INT8 weights, activations quantized on the fly."""
    torch.backends.quantized.engine = _quant_engine()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


LOADERS = {
    "lexicon": Lexicon,
    "tfidf": TfidfLR,
    "w2v_ffn": W2VFFN,
    "bert_fp32": lambda: Bert(int8=False),
    "bert_int8": lambda: Bert(int8=True),
}


def available():
    return [k for k, ps in PATHS.items() if all(Path(p).exists() for p in ps)]


def load(key):
    return LOADERS[key]()


def ensure_weights():
    """Download artifacts from the Hugging Face Hub if they are not present locally."""
    if len(available()) == len(PATHS) or not config.HF_REPO_ID:
        return
    from huggingface_hub import snapshot_download
    print(f"Downloading weights from https://huggingface.co/{config.HF_REPO_ID} ...")
    snapshot_download(repo_id=config.HF_REPO_ID, local_dir=str(A),
                      token=os.environ.get("HF_TOKEN"))
