"""Fill the slide template with YOUR measured results.

  python fill_slides.py --name "Ad Soyad" --id 123456 --repo https://github.com/... --video https://youtu.be/...

Reads results/results.csv, results/errors_bert_int8.csv and data/splits/*.csv (all produced by
train_all.py / evaluate.py) and writes presentation.pptx. Every number on the slides therefore comes
from your own run. Open presentation.pptx in PowerPoint / Google Slides / Keynote and export it as PDF.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData

from src import config as C
from src.lexicon import SEED_LEXICON, LexiconMatcher
from src.normalize import tokenize

TEMPLATE = C.ROOT / "docs" / "slides_template.pptx"
KEYS = ["lexicon", "tfidf", "w2v_ffn", "bert_fp32", "bert_int8"]


def mask(text: str, matcher: LexiconMatcher) -> str:
    out = []
    for word in re.split(r"(\s+)", text):
        if word.strip() and any(matcher.match_token(t) for t in tokenize(word)):
            out.append(word[0] + "*" * (len(word) - 1))
        else:
            out.append(word)
    return "".join(out)


def short(text: str, n: int = 130) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def fmt_mb(x: float) -> str:
    return f"{x:.1f}" if x >= 1 else ("<0.01" if x < 0.01 else f"{x:.2f}")


def findings(r: dict) -> list[str]:
    f1 = {k: r[k]["macro_F1"] for k in r}
    lat = {k: r[k]["latency_ms_mean"] for k in r}
    size = {k: r[k]["size_MB"] for k in r}
    names = {"lexicon": "the lexicon rules", "tfidf": "tf-idf + LR", "w2v_ffn": "word2vec + FFN",
             "bert_fp32": "BERTurk FP32", "bert_int8": "BERTurk INT8"}
    best = max(f1, key=f1.get)
    worst = min(f1, key=f1.get)
    best_classical = max(("lexicon", "tfidf"), key=lambda k: f1[k])
    bert_best = max(("bert_fp32", "bert_int8"), key=lambda k: f1[k])
    out = []
    # 1 accuracy
    gap = f1[bert_best] - f1[best_classical]
    if gap > 0:
        out.append(f"{names[best]} has the highest macro-F1 ({f1[best]:.3f}). The Transformer beats the best classical "
                   f"model ({names[best_classical]}, {f1[best_classical]:.3f}) by {gap * 100:.1f} points. Lowest: {names[worst]} ({f1[worst]:.3f}).")
    else:
        out.append(f"{names[best]} has the highest macro-F1 ({f1[best]:.3f}); the Transformer does not beat "
                   f"{names[best_classical]} on this test set (gap {gap * 100:+.1f} points). Lowest: {names[worst]} ({f1[worst]:.3f}).")
    # 2 speed & size
    fastest = min(lat, key=lat.get)
    out.append(f"Fastest: {names[fastest]} ({lat[fastest]:.2f} ms/message). tf-idf + LR needs {lat['tfidf']:.2f} ms and "
               f"{fmt_mb(size['tfidf'])} MB, BERTurk FP32 {lat['bert_fp32']:.1f} ms and {fmt_mb(size['bert_fp32'])} MB: "
               f"the accuracy gain of the Transformer costs one to two orders of magnitude in latency and size.")
    # 3 quantization
    d = f1["bert_int8"] - f1["bert_fp32"]
    sp = lat["bert_fp32"] / max(lat["bert_int8"], 1e-6)
    sz = size["bert_fp32"] / max(size["bert_int8"], 1e-6)
    ok = "meets" if lat["bert_int8"] <= C.LATENCY_BUDGET_MS else "misses"
    out.append(f"INT8 is {sz:.1f}× smaller and {sp:.1f}× faster than FP32 with a macro-F1 change of {d * 100:+.1f} points. "
               f"At {lat['bert_int8']:.1f} ms/message it {ok} the {C.LATENCY_BUDGET_MS:.0f} ms budget.")
    # 4 classical vs neural
    a, b = f1["w2v_ffn"], f1["tfidf"]
    if a >= b:
        out.append(f"word2vec + FFN ({a:.3f}) beats tf-idf + LR ({b:.3f}): dense embeddings group spelling variants and "
                   f"related words. The lexicon is fast and explainable but brittle ({f1['lexicon']:.3f}).")
    else:
        out.append(f"word2vec + FFN ({a:.3f}) does not beat tf-idf + LR ({b:.3f}): averaging word vectors loses word order, "
                   f"while character n-grams handle noisy spelling. The lexicon is fast but brittle ({f1['lexicon']:.3f}).")
    return [t[0].upper() + t[1:] for t in out]


def build_values(args):
    res_path = C.RESULTS_DIR / "results.csv"
    if not res_path.exists():
        sys.exit("results/results.csv not found – run `python train_all.py` (or `python evaluate.py`) first.")
    res = pd.read_csv(res_path)
    r = {row["key"]: row for _, row in res.iterrows()}
    missing = [k for k in KEYS if k not in r]
    if missing:
        sys.exit(f"results.csv has no rows for {missing} – run the full pipeline first.")

    from src.data import load_splits
    sp = load_splits()
    hw = ""
    md = C.RESULTS_DIR / "results.md"
    if md.exists():
        m = re.search(r"Latency hardware: (.*)", md.read_text(encoding="utf-8"))
        hw = m.group(1).strip() if m else ""

    v = {"NAME": args.name, "STUDENT_ID": args.id, "REPO_URL": args.repo, "VIDEO_URL": args.video,
         "N_TRAIN": f"{len(sp['train']):,}", "N_DEV": f"{len(sp['dev']):,}", "N_TEST": f"{len(sp['test']):,}",
         "TEST_OFF_PCT": f"{sp['test'].label.mean():.1%}", "HARDWARE": hw}
    for k in KEYS:
        v[f"{k}_F1"] = f"{r[k]['macro_F1']:.3f}"
        v[f"{k}_OFFF1"] = f"{r[k]['OFF_F1']:.3f}"
        v[f"{k}_SIZE"] = fmt_mb(r[k]["size_MB"])
        v[f"{k}_LAT"] = f"{r[k]['latency_ms_mean']:.2f}"
        v[f"{k}_P95"] = f"{r[k]['latency_ms_p95']:.2f}"
    v.update({"FP32_SIZE": f"{r['bert_fp32']['size_MB']:.0f}", "INT8_SIZE": f"{r['bert_int8']['size_MB']:.0f}",
              "FP32_LAT": f"{r['bert_fp32']['latency_ms_mean']:.1f}", "INT8_LAT": f"{r['bert_int8']['latency_ms_mean']:.1f}",
              "FP32_F1": v["bert_fp32_F1"], "INT8_F1": v["bert_int8_F1"]})
    li = r["bert_int8"]["latency_ms_mean"]
    v["BUDGET_STATUS"] = (f"met – {li:.1f} ms mean per message on CPU" if li <= C.LATENCY_BUDGET_MS
                          else f"not met – {li:.1f} ms mean per message on CPU (see Next steps)")
    for i, t in enumerate(findings(r), 1):
        v[f"FINDING_{i}"] = t

    # failure examples (most confident mistakes), offensive words masked
    matcher = LexiconMatcher.load(C.LEXICON_PATH) if C.LEXICON_PATH.exists() else LexiconMatcher(SEED_LEXICON)
    err_path = C.RESULTS_DIR / "errors_bert_int8.csv"
    err = pd.read_csv(err_path) if err_path.exists() else pd.DataFrame(columns=["text", "label", "bert_int8"])
    for kind, lab, key in (("false positive", 0, "FP"), ("false negative", 1, "FN")):
        e = err[err.label == lab]
        clean = e[[not matcher.matches(str(t)) for t in e["text"]]]   # avoid showing profanity on slides
        e = clean if len(clean) else e
        if len(e):
            row = e.iloc[0]
            p = row["bert_int8"]
            v[f"{key}_TEXT"] = "“" + short(mask(str(row["text"]), matcher)) + "”"
            v[f"{key}_CONF"] = f"{(p if lab == 0 else 1 - p):.2f}"
        else:
            v[f"{key}_TEXT"] = f"(no {kind} on the test set)"
            v[f"{key}_CONF"] = "–"
    chart = [r[k]["macro_F1"] for k in KEYS]
    return v, chart


def fill(template: Path, out: Path, values: dict, chart_values=None):
    prs = Presentation(str(template))
    pat = re.compile(r"\{\{(\w+)\}\}")

    def repl(run):
        if "{{" in run.text:
            run.text = pat.sub(lambda m: str(values.get(m.group(1), m.group(0))), run.text)

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6:  # group
                walk(sh.shapes)
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for run in p.runs:
                        repl(run)
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        for p in cell.text_frame.paragraphs:
                            for run in p.runs:
                                repl(run)
            if chart_values is not None and getattr(sh, "has_chart", False) and sh.has_chart:
                cd = CategoryChartData(number_format="0.000")
                cd.categories = ["Lexicon", "tf-idf", "w2v+FFN", "BERT FP32", "BERT INT8"]
                cd.add_series("Macro-F1", chart_values)
                sh.chart.replace_data(cd)

    for slide in prs.slides:
        walk(slide.shapes)
    prs.save(str(out))
    left = set()
    for slide in Presentation(str(out)).slides:
        for sh in slide.shapes:
            if sh.has_text_frame:
                left |= set(pat.findall(sh.text_frame.text))
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        left |= set(pat.findall(cell.text))
    return left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="presentation.pptx")
    a = ap.parse_args()
    values, chart = build_values(a)
    left = fill(TEMPLATE, C.ROOT / a.out, values, chart)
    print(f"Saved {a.out}" + (f"  (unfilled fields: {sorted(left)})" if left else "  – all fields filled."))
    print("Open it in PowerPoint / Google Slides and export as PDF for the submission.")


if __name__ == "__main__":
    main()
