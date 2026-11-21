"""Step 1 – classical baselines (Area B): lexicon rule matcher and tf-idf + logistic regression."""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config as C
from src.data import load_splits
from src.lexicon import LexiconMatcher, build_lexicon
from src.normalize import normalize


def main():
    d = load_splits()
    tr, dv = d["train"], d["dev"]

    # ---- rule baseline: lexicon (seed + learned from TRAIN only) with edit-distance matching
    lexicon = build_lexicon(tr.text, tr.label)
    LexiconMatcher.save(lexicon, C.LEXICON_PATH)
    m = LexiconMatcher(lexicon)
    pred = np.array([m.score(t) >= 0.5 for t in dv.text], dtype=int)
    print(f"[lexicon] {len(lexicon)} entries | dev macro-F1 = {f1_score(dv.label, pred, average='macro'):.4f}")

    # ---- tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression; C chosen on dev
    x_tr, x_dv = [normalize(t) for t in tr.text], [normalize(t) for t in dv.text]
    best = (-1, None)
    for c in (0.5, 1, 2, 4, 8):
        pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                         max_features=200_000)),
            ])),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000)),
        ])
        pipe.fit(x_tr, tr.label)
        f1 = f1_score(dv.label, pipe.predict(x_dv), average="macro")
        print(f"[tfidf] C={c:<4} dev macro-F1 = {f1:.4f}")
        if f1 > best[0]:
            best = (f1, pipe)
    joblib.dump(best[1], C.TFIDF_PATH)
    print(f"[tfidf] saved best model (dev macro-F1 = {best[0]:.4f}) -> {C.TFIDF_PATH}")


if __name__ == "__main__":
    main()
