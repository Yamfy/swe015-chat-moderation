"""Real-time chat moderation demo (Gradio).

Each new chat message is labelled live (clean / offensive) with a confidence score by the
selected model - by default the INT8-quantized BERTurk. Flagged messages are masked.
The side panel logs the response time of every message and compares all models.

Run:  python app.py            (add --share on Colab to get a public link)
"""
import argparse
import inspect
import time

import gradio as gr
import pandas as pd

from src import config
from src.lexicon import LexiconMatcher
from src.models import available, ensure_weights, load
from src.normalize import normalize

ensure_weights()
KEYS = available()
assert "bert_int8" in KEYS, "Quantized model not found - run the training scripts or set SWE015_HF_REPO"
print("Loading models:", KEYS)
MODELS = {k: load(k) for k in KEYS}
NAMES = {MODELS[k].name: k for k in KEYS}
DEFAULT = MODELS["bert_int8"].name
LEX = LexiconMatcher.load(config.ART_DIR / "lexicon.json") if "lexicon" in KEYS else None
for m in MODELS.values():                  # warm-up so the first demo message is not slow
    m.predict_proba(["merhaba"])


def mask(text: str) -> str:
    """Mask words that match the offensive lexicon (edit distance <= 1); else hide the whole message."""
    words, hit = text.split(), False
    out = []
    for w in words:
        n = normalize(w)
        if LEX and n and LEX.match(n.split()[0]):
            out.append(w[0] + "*" * (len(w) - 1)); hit = True
        else:
            out.append(w)
    return " ".join(out) if hit else "█" * min(len(text), 24) + "  (message hidden)"


def moderate(message, history, log, model_name):
    if not message or not message.strip():
        return "", history, log, gr.update(), gr.update(), gr.update()
    key = NAMES[model_name]
    t0 = time.perf_counter()
    p = float(MODELS[key].predict_proba([message])[0])
    ms = (time.perf_counter() - t0) * 1000
    offensive = p >= 0.5
    conf = p if offensive else 1 - p
    tag = f"🚫 OFFENSIVE {conf:.0%}" if offensive else f"✅ clean {conf:.0%}"
    shown = mask(message) if offensive else message
    history = history + [{"role": "user", "content": shown},
                         {"role": "assistant", "content": f"{tag} · {MODELS[key].precision} · {ms:.1f} ms"}]
    row = {"#": len(log) + 1, "model": key, "label": "offensive" if offensive else "clean",
           "confidence": round(conf, 3), "latency_ms": round(ms, 2),
           f"≤{config.LATENCY_BUDGET_MS:.0f} ms": "✓" if ms <= config.LATENCY_BUDGET_MS else "✗",
           "message": message[:60]}
    log = log + [row]
    df = pd.DataFrame(log)
    summary = df.groupby("model")["latency_ms"].agg(["count", "mean", "max"]).round(2).reset_index()
    # side-by-side comparison of every model on this message
    comp = []
    for k, m in MODELS.items():
        t1 = time.perf_counter(); pk = float(m.predict_proba([message])[0])
        comp.append({"model": m.name, "precision": m.precision, "P(offensive)": round(pk, 3),
                     "label": "offensive" if pk >= .5 else "clean",
                     "latency_ms": round((time.perf_counter() - t1) * 1000, 2)})
    return "", history, log, df, summary, pd.DataFrame(comp)


chat_kwargs = {"height": 460, "label": "Game chat"}
if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    chat_kwargs["type"] = "messages"

with gr.Blocks(title="SWE015 · Real-time Chat Moderation") as demo:
    gr.Markdown("## 🎮 Real-time Chat Moderation (Turkish) — SWE015 Use Case 4\n"
                f"Each message is classified live on **CPU**. Default model: **{DEFAULT}**. "
                f"Latency budget: **{config.LATENCY_BUDGET_MS:.0f} ms / message**.")
    log_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            model_dd = gr.Dropdown(list(NAMES), value=DEFAULT, label="Model")
            chat = gr.Chatbot(**chat_kwargs)
            with gr.Row():
                box = gr.Textbox(placeholder="Mesaj yaz... / type a message and press Enter",
                                 show_label=False, scale=5)
                send = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(["Harika bir maçtı, tebrikler!", "s1kt1r git buradan ger1zekalı",
                         "bu takım çok kötü oynadı ama olsun", "Seni oyunda bekliyorum kanka"],
                        inputs=box)
        with gr.Column(scale=2):
            gr.Markdown("### ⏱ Response-time log")
            log_df = gr.Dataframe(interactive=False, wrap=True)
            sum_df = gr.Dataframe(label="Latency per model (ms)", interactive=False)
            gr.Markdown("### 🔍 All models on the last message")
            comp_df = gr.Dataframe(interactive=False, wrap=True)

    ins = [box, chat, log_state, model_dd]
    outs = [box, chat, log_state, log_df, sum_df, comp_df]
    box.submit(moderate, ins, outs)
    send.click(moderate, ins, outs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public link (useful on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
