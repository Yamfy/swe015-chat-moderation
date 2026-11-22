"""Area C: fine-tune BERTurk (dbmdz/bert-base-turkish-cased) with a classification head."""
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src import config
from src.data import load_splits
from src.normalize import normalize_for_bert

EPOCHS, BATCH, LR = 3, 32, 2e-5
torch.manual_seed(config.SEED)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

train, val, _ = load_splits()
tok = AutoTokenizer.from_pretrained(config.BERT_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    config.BERT_NAME, num_labels=2, id2label=dict(enumerate(config.LABELS)),
    label2id={l: i for i, l in enumerate(config.LABELS)}).to(device)


def collate(batch):
    texts, labels = zip(*batch)
    enc = tok([normalize_for_bert(t) for t in texts], truncation=True,
              max_length=config.MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    return enc


tr_dl = DataLoader(list(zip(train.text, train.label)), batch_size=BATCH, shuffle=True, collate_fn=collate)
va_dl = DataLoader(list(zip(val.text, val.label)), batch_size=128, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * len(tr_dl)
sched = get_linear_schedule_with_warmup(opt, math.ceil(0.1 * steps), steps)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def evaluate():
    model.eval()
    preds = []
    for b in va_dl:
        b = {k: v.to(device) for k, v in b.items()}
        preds.append(model(**b).logits.argmax(-1).cpu().numpy())
    return f1_score(val.label, np.concatenate(preds), average="macro")


best = -1
for ep in range(EPOCHS):
    model.train(); t0 = time.time()
    for step, b in enumerate(tr_dl):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(**b).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 200 == 0:
            print(f"epoch {ep + 1} step {step}/{len(tr_dl)} loss={loss.item():.4f}")
    f1 = evaluate()
    print(f"epoch {ep + 1}: val macro-F1={f1:.4f} ({time.time() - t0:.0f}s)")
    if f1 > best:
        best = f1
        model.save_pretrained(config.ART_DIR / "bert_fp32", safe_serialization=True)
        tok.save_pretrained(config.ART_DIR / "bert_fp32")
        print("  saved best checkpoint")
print(f"Best val macro-F1 = {best:.4f}")
