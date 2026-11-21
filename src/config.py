"""Central configuration: paths, model names and hyper-parameters."""
import os
from pathlib import Path

# Quick test mode (python train_all.py --quick): small data subset, 1 BERT epoch, separate output
# folders (models_quick/, results_quick/) so that the real results are never mixed with the test run.
QUICK = os.environ.get("SWE015_QUICK") == "1"
_SFX = "_quick" if QUICK else ""

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / f"models{_SFX}"
RESULTS_DIR = ROOT / f"results{_SFX}"

# Dataset: OffensEval-TR 2020 (Çöltekin, LREC 2020), CC-BY
DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
SEED = 42
DEV_FRACTION = 0.10          # dev split carved out of the official training set
LABELS = ["NOT", "OFF"]      # 0 = clean, 1 = offensive

# Transformer (Area C)
BERT_NAME = "dbmdz/bert-base-turkish-cased"   # BERTurk
MAX_LEN = 64
BERT_EPOCHS = 2
BERT_LR = 2e-5
BERT_BATCH = 32

# word2vec + feed-forward net (Area A)
W2V_DIM = 100
FFN_HIDDEN = 128
FFN_EPOCHS = 30
FFN_LR = 0.05

# Efficiency (Area D)
LATENCY_BUDGET_MS = 50.0     # per message, CPU only
LATENCY_RUNS = 200           # >= 100 runs required by the homework

if QUICK:
    BERT_EPOCHS, FFN_EPOCHS, LATENCY_RUNS = 1, 3, 100
    QUICK_SIZES = {"train": 2000, "dev": 300, "test": 300}

# Artifact paths
LEXICON_PATH = MODELS_DIR / "lexicon.json"
TFIDF_PATH = MODELS_DIR / "tfidf_lr.joblib"
W2V_PATH = MODELS_DIR / "w2v" / "word2vec.model"
FFN_PATH = MODELS_DIR / "w2v" / "ffn.pt"
BERT_FP32_DIR = MODELS_DIR / "bert_fp32"
BERT_INT8_PATH = MODELS_DIR / "bert_int8" / "bert_int8_state_dict.pt"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
