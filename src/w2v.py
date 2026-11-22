"""Area A - our own word2vec (skip-gram with negative sampling) + feedforward classifier.

Both networks are written in PyTorch and trained with SGD + backpropagation.
word2vec is trained ONLY on the training messages (no pretrained vectors).
"""
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalize import tokenize


# ---------------------------------------------------------------- word2vec
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)    # target-word vectors (the embeddings we keep)
        self.out = nn.Embedding(vocab_size, dim)    # context-word vectors
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                                   # (B, d)
        u_pos = self.out(context)                              # (B, d)
        u_neg = self.out(negatives)                            # (B, k, d)
        pos = F.logsigmoid((v * u_pos).sum(-1))                # log σ(u_o · v_c)
        neg = F.logsigmoid(-torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)).sum(-1)  # Σ log σ(-u_k · v_c)
        return -(pos + neg).mean()


def build_vocab(token_lists, min_count=2):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = ["<unk>"] + [w for w, c in counts.most_common() if c >= min_count]
    return {w: i for i, w in enumerate(itos)}, counts


def train_word2vec(texts, dim=100, window=5, negatives=5, epochs=5, lr=0.005,
                   batch_size=4096, device="cpu", seed=42, log=print):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    token_lists = [tokenize(t) for t in texts]
    stoi, counts = build_vocab(token_lists)
    V = len(stoi)
    # unigram^0.75 distribution for negative sampling (Mikolov et al., 2013)
    freq = np.array([counts.get(w, 1) for w in stoi], dtype=np.float64) ** 0.75
    noise = torch.tensor(freq / freq.sum(), dtype=torch.float)
    # (center, context) pairs with a dynamic window
    pairs = []
    for toks in token_lists:
        ids = [stoi[t] for t in toks if t in stoi]
        for i, c in enumerate(ids):
            w = random.randint(1, window)
            for j in range(max(0, i - w), min(len(ids), i + w + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    pairs = torch.tensor(pairs, dtype=torch.long)
    log(f"word2vec: vocab={V}, training pairs={len(pairs)}")
    model = SkipGramNS(V, dim).to(device)
    # Adam: plain SGD on the batch-mean loss barely moved the vectors (loss stayed at 6*ln2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(pairs))
        total = 0.0
        for k in range(0, len(pairs), batch_size):
            b = pairs[perm[k:k + batch_size]]
            neg = torch.multinomial(noise, len(b) * negatives, replacement=True).view(len(b), negatives)
            loss = model(b[:, 0].to(device), b[:, 1].to(device), neg.to(device))
            opt.zero_grad(); loss.backward(); opt.step()        # backpropagation + SGD step
            total += loss.item() * len(b)
        log(f"  epoch {ep + 1}/{epochs}  loss={total / len(pairs):.4f}")
    emb = model.inp.weight.detach().cpu().numpy()
    return stoi, emb


def most_similar(word, stoi, emb, k=8):
    """Cosine similarity neighbours - handy for the slides / sanity check."""
    if word not in stoi:
        return []
    itos = {i: w for w, i in stoi.items()}
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = e @ e[stoi[word]]
    return [(itos[i], float(sims[i])) for i in np.argsort(-sims)[1:k + 1]]


# ---------------------------------------------------------------- classifier
class FFN(nn.Module):
    """Averaged word2vec -> standardisation -> Linear -> ReLU -> Dropout -> Linear (2 classes).

    The averaged SGNS vectors are tiny and nearly identical across messages, so the input is
    standardised with the TRAINING-set mean / std (stored as buffers, saved with the weights).
    """

    def __init__(self, dim=100, hidden=128, dropout=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 2))

    def fit_scaler(self, x):
        self.mu.copy_(x.mean(0))
        self.sigma.copy_(x.std(0) + 1e-6)

    def forward(self, x):
        return self.net((x - self.mu) / self.sigma)


def average_embeddings(texts, stoi, emb):
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for n, t in enumerate(texts):
        ids = [stoi[w] for w in tokenize(t) if w in stoi]
        if ids:
            out[n] = emb[ids].mean(0)
    return out


def save_w2v(stoi, emb, vocab_path, emb_path):
    json.dump(stoi, open(vocab_path, "w", encoding="utf-8"), ensure_ascii=False)
    np.save(emb_path, emb.astype(np.float32))


def load_w2v(vocab_path, emb_path):
    return json.load(open(vocab_path, encoding="utf-8")), np.load(emb_path)
