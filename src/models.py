"""Uniform inference wrappers for the four rungs of the model ladder (+ BERT FP32).

Every model exposes:
    name           display name
    predict_proba(texts) -> np.ndarray of P(offensive) for each text
    size_mb()      size of the model artifacts on disk
These same classes are used by evaluate.py and by app.py, so the numbers in the
results table come from exactly the code the application runs.
"""
import os
from pathlib import Path

import numpy as np

from . import config as C
from .normalize import light_normalize, normalize, tokenize


def _dir_size_mb(*paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total / 1e6


# ---------------------------------------------------------------- 1. rule baseline (B)
class LexiconModel:
    name = "Lexicon + edit distance (rules)"
    short = "lexicon"

    def __init__(self):
        from .lexicon import LexiconMatcher
        self.matcher = LexiconMatcher.load(C.LEXICON_PATH)

    def predict_proba(self, texts):
        return np.array([self.matcher.score(t) for t in texts])

    def size_mb(self):
        return _dir_size_mb(C.LEXICON_PATH)


# ---------------------------------------------------------------- 2. classical baseline (B)
class TfidfModel:
    name = "tf-idf + Logistic Regression"
    short = "tfidf"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(C.TFIDF_PATH)

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1]

    def size_mb(self):
        return _dir_size_mb(C.TFIDF_PATH)


# ---------------------------------------------------------------- 3. neural network (A)
def build_ffn(in_dim: int, hidden: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 2))


def message_vectors(token_lists, kv, dim):
    """Mean- and max-pooled word2vec vectors -> [N, 2*dim]."""
    out = np.zeros((len(token_lists), 2 * dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [kv[t] for t in toks if t in kv]
        if vecs:
            v = np.stack(vecs)
            out[i, :dim] = v.mean(0)
            out[i, dim:] = v.max(0)
    return out


class W2VFFNModel:
    name = "word2vec + feed-forward NN"
    short = "w2v_ffn"

    def __init__(self):
        import torch
        from gensim.models import Word2Vec
        self.torch = torch
        self.kv = Word2Vec.load(str(C.W2V_PATH)).wv
        ckpt = torch.load(C.FFN_PATH, map_location="cpu")
        self.net = build_ffn(ckpt["in_dim"], ckpt["hidden"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    def predict_proba(self, texts):
        x = message_vectors([tokenize(t) for t in texts], self.kv, self.kv.vector_size)
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(x))
        return self.torch.softmax(logits, -1)[:, 1].numpy()

    def size_mb(self):
        return _dir_size_mb(C.W2V_PATH.parent)


# ---------------------------------------------------------------- 4. Transformer (C) and 5. quantized (D)
class _BertBase:
    def _setup(self, device, model_dir):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)

    def predict_proba(self, texts, batch_size=32):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tok([light_normalize(t) for t in texts[i:i + batch_size]], padding=True,
                               truncation=True, max_length=C.MAX_LEN, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])


class BertFP32Model(_BertBase):
    name = "BERTurk fine-tuned (FP32)"
    short = "bert_fp32"

    def __init__(self, device="cpu"):
        from transformers import AutoModelForSequenceClassification
        self._setup(device, C.BERT_FP32_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(C.BERT_FP32_DIR).to(device).eval()

    def size_mb(self):
        return _dir_size_mb(*[p for p in C.BERT_FP32_DIR.glob("*") if p.suffix in (".safetensors", ".bin")])


def best_device() -> str:
    """cuda (NVIDIA / Colab) > mps (Apple Silicon Mac) > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_quant_engine():
    """INT8 kernel backend: qnnpack on ARM (Apple Silicon), x86/fbgemm on Intel/AMD."""
    import platform
    import torch
    engines = torch.backends.quantized.supported_engines
    if platform.machine().lower() in ("arm64", "aarch64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    else:
        for e in ("x86", "fbgemm", "qnnpack"):
            if e in engines:
                torch.backends.quantized.engine = e
                break


def quantize_bert(model):
    """Post-training dynamic INT8 quantization of all nn.Linear layers (CPU)."""
    import warnings
    import torch
    setup_quant_engine()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:  # older torch
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


class BertInt8Model(_BertBase):
    name = "BERTurk dynamic INT8 (quantized)"
    short = "bert_int8"

    def __init__(self):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification
        int8_dir = C.BERT_INT8_PATH.parent          # self-contained: config + tokenizer + INT8 weights
        self._setup("cpu", int8_dir)
        # build the architecture from the config, quantize it, then load the INT8 weights
        cfg = AutoConfig.from_pretrained(int8_dir)
        skeleton = AutoModelForSequenceClassification.from_config(cfg)
        self.model = quantize_bert(skeleton)
        state = torch.load(C.BERT_INT8_PATH, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

    def size_mb(self):
        return _dir_size_mb(C.BERT_INT8_PATH)


ALL_MODELS = [LexiconModel, TfidfModel, W2VFFNModel, BertFP32Model, BertInt8Model]


def load_all(skip_missing=True, bert_device="cpu"):
    models = {}
    for cls in ALL_MODELS:
        try:
            models[cls.short] = cls(device=bert_device) if cls is BertFP32Model else cls()
        except Exception as e:  # noqa: BLE001
            if not skip_missing:
                raise
            print(f"[warn] could not load {cls.name}: {e}")
    return models
