"""Evaluate every model on the SAME held-out test set (official OffensEval-TR 2020 test).

Outputs (results/):
  results.md / results.csv / results.json  - metric, size (MB), CPU latency (mean & p95 over N runs)
  errors_bert_int8.csv                     - misclassified test messages (for the failure-example slide)
  hardware.txt                             - where the latency was measured
"""
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src import config
from src.data import load_splits
from src.models import PATHS, available, load, size_mb

torch.set_num_threads(max(1, torch.get_num_threads()))
_, _, test = load_splits()
texts, y = test.text.tolist(), test.label.values


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return (f"{cpu} | {platform.system()} {platform.release()} | torch threads={torch.get_num_threads()} "
            f"| Python {platform.python_version()} | torch {torch.__version__} | CPU only")


def latency_ms(model, n=config.LATENCY_RUNS):
    """Per-message latency: one message at a time, end-to-end (normalisation + model), CPU."""
    sample = [texts[i % len(texts)] for i in range(n + 10)]
    for t in sample[:10]:                                # warm-up
        model.predict_proba([t])
    times = []
    for t in sample[10:]:
        t0 = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


rows, probs = [], {}
order = ["lexicon", "tfidf", "w2v_ffn", "bert_fp32", "bert_int8"]
for key in [k for k in order if k in available()]:
    print(f"\n=== {key} ===")
    m = load(key)
    t0 = time.time()
    p = m.predict_proba(texts)
    probs[key] = p
    pred = (p >= 0.5).astype(int)
    lat, p95 = latency_ms(m)
    rows.append({
        "model": m.name, "key": key, "precision": m.precision,
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "f1_offensive": round(f1_score(y, pred), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "size_mb": round(size_mb(PATHS[key]), 2),
        "latency_ms_mean": round(lat, 2), "latency_ms_p95": round(p95, 2),
        "meets_budget": lat <= config.LATENCY_BUDGET_MS,
    })
    print(classification_report(y, pred, target_names=config.LABELS, digits=4))
    print(f"latency: mean {lat:.2f} ms, p95 {p95:.2f} ms  (test-set pass {time.time() - t0:.0f}s)")

df = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(config.RESULTS_DIR / "results.csv", index=False)
hw = hardware()
(config.RESULTS_DIR / "hardware.txt").write_text(hw + "\n")

extra = {}
if "bert_fp32" in probs and "bert_int8" in probs:
    agree = float(((probs["bert_fp32"] >= .5) == (probs["bert_int8"] >= .5)).mean())
    extra["fp32_int8_prediction_agreement"] = round(agree, 4)
if "bert_int8" in probs:
    pred = (probs["bert_int8"] >= .5).astype(int)
    err = test.assign(pred=pred, p_offensive=probs["bert_int8"].round(3))[pred != y]
    err.assign(error=np.where(err.label == 1, "missed offensive (FN)", "false alarm (FP)")) \
       .sort_values("p_offensive").to_csv(config.RESULTS_DIR / "errors_bert_int8.csv", index=False)

json.dump({"hardware": hw, "test_size": len(test), "test_offensive_rate": round(float(y.mean()), 4),
           "latency_runs": config.LATENCY_RUNS, "latency_budget_ms": config.LATENCY_BUDGET_MS,
           "models": rows, **extra},
          open(config.RESULTS_DIR / "results.json", "w"), indent=2)

md = ["| Model | Precision | Macro-F1 | F1 (offensive) | Size (MB) | Latency mean / p95 (ms) |",
      "|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['model']} | {r['precision']} | {r['macro_f1']:.4f} | {r['f1_offensive']:.4f} | "
              f"{r['size_mb']:.2f} | {r['latency_ms_mean']:.2f} / {r['latency_ms_p95']:.2f} |")
md.append(f"\nTest set: official OffensEval-TR 2020 test, n={len(test)} ({y.mean():.1%} offensive). "
          f"Latency: {config.LATENCY_RUNS} single-message runs, {hw}.")
if extra:
    md.append(f"FP32 vs INT8 prediction agreement: {extra['fp32_int8_prediction_agreement']:.2%}")
(config.RESULTS_DIR / "results.md").write_text("\n".join(md) + "\n")
print("\n" + "\n".join(md))
