"""Area A: train our own word2vec on the training messages, then a feedforward net (SGD + backprop)."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src import config
from src.data import load_splits
from src.w2v import FFN, average_embeddings, most_similar, save_w2v, train_word2vec

torch.manual_seed(config.SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
train, val, _ = load_splits()

# ---- 1. word2vec (skip-gram, negative sampling) on training messages only
stoi, emb = train_word2vec(train.text.tolist(), dim=100, epochs=5, device=device)
save_w2v(stoi, emb, config.ART_DIR / "w2v_vocab.json", config.ART_DIR / "w2v_emb.npy")
for probe in ("mac", "hakem", "aptal", "gerizekali"):
    print(f"  nearest to '{probe}':", [w for w, _ in most_similar(probe, stoi, emb, 6)])

# ---- 2. feedforward network on averaged embeddings
Xtr = torch.from_numpy(average_embeddings(train.text.tolist(), stoi, emb))
Xva = torch.from_numpy(average_embeddings(val.text.tolist(), stoi, emb))
ytr = torch.tensor(train.label.values)
net = FFN(dim=emb.shape[1])
# class weights for imbalance (approx. 1 offensive : 4 clean)
w = torch.tensor(len(ytr) / (2 * np.bincount(ytr.numpy())), dtype=torch.float)
loss_fn = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)

best_f1, best_state = -1, None
for epoch in range(40):
    net.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]
        loss = loss_fn(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()      # backpropagation + SGD update
    net.eval()
    with torch.no_grad():
        f1 = f1_score(val.label, net(Xva).argmax(-1).numpy(), average="macro")
    if f1 > best_f1:
        best_f1, best_state = f1, copy.deepcopy(net.state_dict())
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch + 1:2d}  loss={loss.item():.4f}  val macro-F1={f1:.4f}")

torch.save(best_state, config.ART_DIR / "ffn.pt")
print(f"Best val macro-F1 = {best_f1:.4f} (early-stopping on validation set)")
