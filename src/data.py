"""Download OffensEval-TR 2020 and build fixed train / dev / test splits.

The official test set is used as the single held-out test set for ALL models (R3).
A stratified dev split (10 %) is carved out of the official training set and is used
only for model selection.
"""
import csv
import io
import zipfile
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .config import DATA_DIR, DATA_URL, DEV_FRACTION, SEED

SPLIT_DIR = DATA_DIR / "splits"


def _read_tsv(raw: bytes) -> pd.DataFrame:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE))
    return pd.DataFrame(rows)


def _download() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    names = zf.namelist()
    train_f = next(n for n in names if n.endswith("offenseval-tr-training-v1.tsv"))
    test_f = next(n for n in names if n.endswith("offenseval-tr-testset-v1.tsv"))
    label_f = next(n for n in names if n.endswith("offenseval-tr-labela-v1.tsv"))

    train = _read_tsv(zf.read(train_f))
    test = _read_tsv(zf.read(test_f))
    labels = pd.read_csv(io.BytesIO(zf.read(label_f)), header=None, names=["id", "subtask_a"],
                         dtype=str, quoting=csv.QUOTE_NONE)
    test = test.merge(labels, on="id", how="inner")
    return train, test


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"tweet": "text"})
    df = df[df["subtask_a"].isin(["NOT", "OFF"])].copy()
    df["label"] = (df["subtask_a"] == "OFF").astype(int)
    df["text"] = df["text"].astype(str)
    return df[["id", "text", "label"]].reset_index(drop=True)


def load_splits() -> dict[str, pd.DataFrame]:
    """Return {'train', 'dev', 'test'} DataFrames with columns id, text, label."""
    splits = _load_full()
    if C.QUICK:  # small stratified subsets for the quick test run
        splits = {s: (df if len(df) <= C.QUICK_SIZES[s] else
                      train_test_split(df, train_size=C.QUICK_SIZES[s], random_state=SEED, stratify=df["label"])[0]
                      ).reset_index(drop=True) for s, df in splits.items()}
    return splits


def _load_full() -> dict[str, pd.DataFrame]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {s: SPLIT_DIR / f"{s}.csv" for s in ("train", "dev", "test")}
    if all(p.exists() for p in paths.values()):
        return {s: pd.read_csv(p, dtype={"id": str, "text": str}).fillna({"text": ""}) for s, p in paths.items()}

    train_full, test = _download()
    train_full, test = _clean(train_full), _clean(test)
    train, dev = train_test_split(train_full, test_size=DEV_FRACTION, random_state=SEED,
                                  stratify=train_full["label"])
    splits = {"train": train.reset_index(drop=True), "dev": dev.reset_index(drop=True), "test": test}
    for s, df in splits.items():
        df.to_csv(paths[s], index=False)
    for s, df in splits.items():
        print(f"{s:5s}: {len(df):6d} messages, offensive = {df.label.mean():.1%}")
    return splits


if __name__ == "__main__":
    load_splits()
