"""Step 5 – evaluate every model on the SAME held-out test set (official OffensEval-TR test split).

Writes:
  results/results.md, results/results.csv   – the results table (R4)
  results/predictions.csv                   – P(OFF) of every model for every test message
  results/errors_bert_int8.csv              – misclassified messages (for the failure-example slide)
  results/report.txt                        – classification reports, confusion matrices, hardware
Latency = one message per call (batch size 1) on CPU, averaged over LATENCY_RUNS runs after warm-up.
"""
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config as C
from src.data import load_splits
from src.models import ALL_MODELS, BertFP32Model, best_device


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return f"{cpu} | {os.cpu_count()} logical cores | torch threads = {torch.get_num_threads()} | {platform.system()}"


def latency_ms(model, texts, runs=C.LATENCY_RUNS, warmup=10):
    for t in texts[:warmup]:
        model.predict_proba([t])
    times = []
    for i in range(runs):
        t = texts[i % len(texts)]
        s = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - s) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    test = load_splits()["test"]
    texts, y = test.text.tolist(), test.label.values
    rng = np.random.default_rng(C.SEED)
    lat_texts = [texts[i] for i in rng.choice(len(texts), size=C.LATENCY_RUNS, replace=False)]
    gpu = best_device()

    rows, preds, report = [], {"id": test.id, "text": texts, "label": y}, [f"Hardware (latency): {hardware()}\n"]
    for cls in ALL_MODELS:
        print(f"\n=== {cls.name} ===")
        # task metric (BERT FP32 may use the GPU for speed – predictions are the same)
        model = cls(device=gpu) if cls is BertFP32Model else cls()
        p = model.predict_proba(texts)
        yhat = (p >= 0.5).astype(int)
        preds[cls.short] = np.round(p, 4)
        # latency always on CPU, one message at a time
        if cls is BertFP32Model and gpu != "cpu":
            model.model.to("cpu"); model.device = "cpu"
        mean_ms, p95_ms = latency_ms(model, lat_texts)
        row = {
            "key": cls.short,
            "model": cls.name,
            "macro_F1": round(f1_score(y, yhat, average="macro"), 4),
            "OFF_F1": round(f1_score(y, yhat, pos_label=1), 4),
            "accuracy": round(accuracy_score(y, yhat), 4),
            "size_MB": round(model.size_mb(), 1),
            "latency_ms_mean": round(mean_ms, 2),
            "latency_ms_p95": round(p95_ms, 2),
            "meets_budget": mean_ms <= C.LATENCY_BUDGET_MS,
        }
        rows.append(row)
        print(row)
        report += [f"\n=== {cls.name} ===", classification_report(y, yhat, target_names=C.LABELS, digits=4),
                   f"confusion matrix [rows=true NOT/OFF, cols=pred NOT/OFF]:\n{confusion_matrix(y, yhat)}"]
        del model

    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS_DIR / "results.csv", index=False)
    pd.DataFrame(preds).to_csv(C.RESULTS_DIR / "predictions.csv", index=False)
    with open(C.RESULTS_DIR / "results.md", "w", encoding="utf-8") as f:
        f.write(f"Test set: official OffensEval-TR 2020 test split, n = {len(test)} "
                f"({y.mean():.1%} offensive)\n\nLatency hardware: {hardware()}\n"
                f"Latency: batch size 1, mean / p95 over {C.LATENCY_RUNS} runs, budget = {C.LATENCY_BUDGET_MS} ms\n\n")
        f.write(res.to_markdown(index=False))
    (C.RESULTS_DIR / "report.txt").write_text("\n".join(report), encoding="utf-8")

    if "bert_int8" in preds:
        df = pd.DataFrame(preds)
        err = df[(df.bert_int8 >= 0.5).astype(int) != df.label].copy()
        err["error"] = np.where(err.label == 1, "false negative", "false positive")
        err["confidence"] = np.abs(err.bert_int8 - 0.5)
        err.sort_values("confidence", ascending=False).to_csv(C.RESULTS_DIR / "errors_bert_int8.csv", index=False)

    print("\n" + res.to_string(index=False))
    print(f"\nSaved to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
