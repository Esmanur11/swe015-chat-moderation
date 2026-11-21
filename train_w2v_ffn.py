"""Step 2 – neural network (Area A + B): our own word2vec trained on the training messages,
feeding a feed-forward network trained with plain SGD and backpropagation."""
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

from src import config as C
from src.data import load_splits
from src.models import build_ffn, message_vectors
from src.normalize import tokenize


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    d = load_splits()
    tr, dv = d["train"], d["dev"]
    tok_tr, tok_dv = [tokenize(t) for t in tr.text], [tokenize(t) for t in dv.text]

    # ---- word2vec (skip-gram) trained ONLY on the training messages
    w2v = Word2Vec(sentences=tok_tr, vector_size=C.W2V_DIM, window=5, min_count=2, sg=1,
                   epochs=15, workers=4, seed=C.SEED)
    C.W2V_PATH.parent.mkdir(parents=True, exist_ok=True)
    w2v.save(str(C.W2V_PATH))
    print(f"[w2v] vocabulary = {len(w2v.wv):,} words")
    for probe in ("siktir", "salak", "mac"):
        if probe in w2v.wv:
            print(f"[w2v] nearest to '{probe}':", [w for w, _ in w2v.wv.most_similar(probe, topn=5)])

    x_tr = torch.from_numpy(message_vectors(tok_tr, w2v.wv, C.W2V_DIM))
    x_dv = torch.from_numpy(message_vectors(tok_dv, w2v.wv, C.W2V_DIM))
    y_tr = torch.tensor(tr.label.values)

    # ---- feed-forward net, SGD + backprop, class-weighted loss (classes are imbalanced)
    net = build_ffn(x_tr.shape[1], C.FFN_HIDDEN)
    pos = y_tr.float().mean().item()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5 / (1 - pos), 0.5 / pos]))
    opt = torch.optim.SGD(net.parameters(), lr=C.FFN_LR, momentum=0.9)
    best_f1, best_state = -1, None
    for epoch in range(1, C.FFN_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
        net.eval()
        with torch.no_grad():
            pred = net(x_dv).argmax(-1).numpy()
        f1 = f1_score(dv.label, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"[ffn] epoch {epoch:2d} loss={loss.item():.4f} dev macro-F1={f1:.4f}")
    torch.save({"state_dict": best_state, "in_dim": x_tr.shape[1], "hidden": C.FFN_HIDDEN}, C.FFN_PATH)
    print(f"[ffn] saved best model (dev macro-F1 = {best_f1:.4f}) -> {C.FFN_PATH}")


if __name__ == "__main__":
    main()
