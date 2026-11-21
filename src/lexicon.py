"""Area B – rule baseline: offensive-word lexicon matched with minimum edit distance.

The lexicon = a small hand-written seed list of Turkish profanity stems
            + words that are strongly associated with the OFF class in the TRAINING split
              (count >= MIN_COUNT and P(OFF | word) >= MIN_PRECISION).
A message is flagged if any token matches a lexicon entry:
  * exact match for short entries (< 4 letters, e.g. "amk", "aq"),
  * otherwise a prefix of the token (Turkish is agglutinative: "siktir" -> "siktirin")
    is within Levenshtein distance 1 of the entry (exact prefix for 4-letter entries).
"""
import json
from collections import Counter
from functools import lru_cache

from .normalize import tokenize

# Hand-written seed stems (already in normalized, ASCII-folded form).
SEED_LEXICON = [
    "amk", "aq", "amq", "mk", "oc", "aw", "sg", "siktir", "sikerim", "sikik", "sikis", "yarrak", "yarak",
    "orospu", "orosbu", "kahpe", "pic", "pezevenk", "gavat", "ibne", "got", "gotveren", "gotlek",
    "amina", "amcik", "yavsak", "serefsiz", "haysiyetsiz", "pust", "dalyarak", "salak", "aptal",
    "gerizekali", "mal", "andaval", "dangalak", "hiyar", "ahmak", "lavuk", "kaltak", "surtuk",
]

MIN_COUNT = 5
MIN_PRECISION = 0.85
MAX_LEARNED = 200


def levenshtein(a: str, b: str) -> int:
    """Minimum edit distance (insert / delete / substitute = 1), dynamic programming."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                # deletion
                           cur[j - 1] + 1,             # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def build_lexicon(train_texts, train_labels) -> list[str]:
    """Seed list + words learned from the training split only."""
    tot, off = Counter(), Counter()
    for text, y in zip(train_texts, train_labels):
        for tok in set(tokenize(text)):
            tot[tok] += 1
            off[tok] += int(y)
    learned = [w for w, c in tot.items()
               if c >= MIN_COUNT and len(w) >= 2 and off[w] / c >= MIN_PRECISION]
    learned.sort(key=lambda w: -tot[w])
    seed = [w.encode("ascii", "ignore").decode() for w in SEED_LEXICON]
    return sorted(set(seed) | set(learned[:MAX_LEARNED]))


ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def edits1(word: str) -> set[str]:
    """All strings within edit distance 1 of `word` (deletions, substitutions, insertions)."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {a + b[1:] for a, b in splits if b}
    subs = {a + c + b[1:] for a, b in splits if b for c in ALPHABET}
    inserts = {a + c + b for a, b in splits for c in ALPHABET}
    return deletes | subs | inserts | {word}


class LexiconMatcher:
    def __init__(self, lexicon: list[str]):
        self.short = {w for w in lexicon if len(w) < 4}
        self.long = [w for w in lexicon if len(w) >= 4]
        # pre-compute the edit-distance-1 neighbourhood of every long entry -> O(1) lookups
        self.neigh: dict[str, str] = {}
        for w in sorted(self.long, key=len):
            for v in (edits1(w) if len(w) >= 5 else {w}):   # 4-letter entries: exact prefix only
                self.neigh.setdefault(v, w)

    @lru_cache(maxsize=200_000)
    def match_token(self, tok: str) -> str | None:
        """Return the lexicon entry a token matches, or None.
        Short entries must match exactly; long entries may match a PREFIX of the token
        (Turkish suffixes: "siktir" -> "siktirin") within Levenshtein distance 1."""
        if tok in self.short:
            return tok
        for k in range(3, len(tok) + 1):
            w = self.neigh.get(tok[:k])
            if w is not None and levenshtein(tok[:k], w) <= 1:
                return w
        return None

    def matches(self, text: str) -> list[tuple[str, str]]:
        """List of (token, lexicon entry) pairs found in a message."""
        out = []
        for tok in tokenize(text):
            m = self.match_token(tok)
            if m:
                out.append((tok, m))
        return out

    def score(self, text: str) -> float:
        """Pseudo-probability of OFF: 0 matches -> 0.0, 1 -> 0.5, 2 -> 0.75, ..."""
        n = len(self.matches(text))
        return 1.0 - 0.5 ** n

    @staticmethod
    def save(lexicon: list[str], path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8")))
