"""Area D: post-training dynamic INT8 quantization of the fine-tuned BERTurk (CPU)."""
import shutil

import torch
from transformers import AutoModelForSequenceClassification

from src import config
from src.models import quantize, size_mb

src_dir, dst_dir = config.ART_DIR / "bert_fp32", config.ART_DIR / "bert_int8"
dst_dir.mkdir(exist_ok=True)
model = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
qmodel = quantize(model)
torch.save(qmodel.state_dict(), dst_dir / "model_int8.pt")
for f in src_dir.iterdir():                      # copy config + tokenizer files
    if f.suffix not in (".safetensors", ".bin"):
        shutil.copy(f, dst_dir / f.name)
print(qmodel.bert.encoder.layer[0].attention.self.query)
print(f"FP32: {size_mb([src_dir]):.1f} MB  ->  INT8: {size_mb([dst_dir]):.1f} MB")
