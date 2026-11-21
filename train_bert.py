"""Step 3 – Transformer (Area C): fine-tune BERTurk with a classification head.
Run on a GPU (e.g. Google Colab T4): ~3-5 minutes per epoch."""
import contextlib

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import config as C
from src.data import load_splits
from src.models import best_device
from src.normalize import light_normalize


def batches(tok, texts, labels, shuffle):
    data = list(zip([light_normalize(t) for t in texts], labels))

    def collate(b):
        enc = tok([x for x, _ in b], padding=True, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b])
        return enc
    return DataLoader(data, batch_size=C.BERT_BATCH, shuffle=shuffle, collate_fn=collate)


def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.inference_mode():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            preds.append(model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1).cpu())
            gold.append(b["labels"].cpu())
    return f1_score(torch.cat(gold), torch.cat(preds), average="macro")


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = best_device()
    print(f"[bert] device = {device}" + ("  (WARNING: very slow without a GPU)" if device == "cpu" else ""))
    d = load_splits()
    tok = AutoTokenizer.from_pretrained(C.BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.BERT_NAME, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}).to(device)

    train_dl = batches(tok, d["train"].text, d["train"].label.tolist(), shuffle=True)
    dev_dl = batches(tok, d["dev"].text, d["dev"].label.tolist(), shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=C.BERT_LR, weight_decay=0.01)
    steps = len(train_dl) * C.BERT_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best = -1.0
    for epoch in range(1, C.BERT_EPOCHS + 1):
        model.train()
        for step, b in enumerate(train_dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            amp = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with amp:
                loss = model(**b).loss
            opt.zero_grad()
            if scaler:                      # mixed precision on NVIDIA GPUs
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:                           # Apple MPS / CPU: plain FP32
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            if step % 200 == 0:
                print(f"[bert] epoch {epoch} step {step}/{len(train_dl)} loss={loss.item():.4f}")
        f1 = evaluate(model, dev_dl, device)
        print(f"[bert] epoch {epoch} dev macro-F1 = {f1:.4f}")
        if f1 > best:
            best = f1
            model.save_pretrained(C.BERT_FP32_DIR)
            tok.save_pretrained(C.BERT_FP32_DIR)
    print(f"[bert] saved best model (dev macro-F1 = {best:.4f}) -> {C.BERT_FP32_DIR}")


if __name__ == "__main__":
    main()
