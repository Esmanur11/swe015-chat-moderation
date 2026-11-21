"""Step 4 – efficient model (Area D): post-training dynamic INT8 quantization of BERTurk.

All nn.Linear layers (attention Q/K/V/output, feed-forward, classifier) get INT8 weights;
activations are quantized on the fly at inference time. Embeddings and LayerNorm stay FP32.
The result runs on CPU only, which matches the 'CPU-only server' scenario."""
import shutil
import warnings

import torch
from transformers import AutoModelForSequenceClassification

from src import config as C
from src.models import quantize_bert


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # torch.ao.quantization is deprecated but still supported
    fp32 = AutoModelForSequenceClassification.from_pretrained(C.BERT_FP32_DIR)
    int8 = quantize_bert(fp32)
    out_dir = C.BERT_INT8_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(int8.state_dict(), C.BERT_INT8_PATH)
    # copy config + tokenizer so the INT8 folder is self-contained
    for f in C.BERT_FP32_DIR.iterdir():
        if f.suffix not in (".safetensors", ".bin"):
            shutil.copy(f, out_dir / f.name)
    n_q = sum(1 for m in int8.modules() if "quantized" in type(m).__module__ and "Linear" in type(m).__name__)
    print(f"[quant] {n_q} Linear layers quantized to INT8 -> {C.BERT_INT8_PATH}")


if __name__ == "__main__":
    main()
