"""Area B – regex normalization of noisy Turkish chat text.

Two levels are provided:
  * light_normalize(): for BERTurk – keeps Turkish characters and casing information,
    only fixes noise that the subword tokenizer handles badly.
  * normalize(): for the classical models (lexicon, tf-idf, word2vec) – lower-cases,
    decodes leetspeak, collapses repeated letters, removes in-word separators and folds
    Turkish characters to ASCII so that "sıçmak" and "sicmak" become the same token.
"""
import re

# Turkish-aware lower-casing (Python's str.lower() maps "I" to "i", which is wrong for Turkish)
_TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})
# Fold Turkish letters to ASCII (people often type without Turkish characters)
_TR_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u"})
# Leetspeak – applied only inside tokens that mix letters and digits/symbols
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s", "€": "e"})

RE_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)", re.IGNORECASE)
RE_USER = re.compile(r"@USER\b|@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_REPEAT = re.compile(r"(\w)\1{2,}")                         # 3+ identical letters
RE_INWORD_SEP = re.compile(r"(?<=\w)[.\-_*+]+(?=\w)")           # s.a.l.a.k / a-p-t-a-l / a*ptal
RE_LEET_TOKEN = re.compile(r"\b(?=[\w@$€]*[a-zçğıöşü])(?=[\w@$€]*[0-9@$€])[\w@$€]+\b", re.IGNORECASE)
RE_NON_WORD = re.compile(r"[^\w\s]")
RE_SPACES = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[a-z0-9]+")


def tr_lower(text: str) -> str:
    return text.translate(_TR_UPPER).lower()


def light_normalize(text: str) -> str:
    """Normalization for the Transformer."""
    text = RE_URL.sub("URL", text)
    text = RE_USER.sub("@USER", text)
    text = RE_REPEAT.sub(r"\1\1", text)                        # "çoooook" -> "çook"
    return RE_SPACES.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Full normalization for the classical / word2vec models."""
    text = RE_URL.sub(" ", text)
    text = RE_USER.sub(" ", text)
    text = RE_HASHTAG.sub(r" \1 ", text)
    text = tr_lower(text)
    text = RE_LEET_TOKEN.sub(lambda m: m.group(0).translate(_LEET), text)
    text = RE_INWORD_SEP.sub("", text)
    text = text.translate(_TR_FOLD)
    text = RE_REPEAT.sub(r"\1", text)                          # "yaaaa" -> "ya"
    text = RE_NON_WORD.sub(" ", text)
    return RE_SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokens of the fully normalized text."""
    return RE_TOKEN.findall(normalize(text))


if __name__ == "__main__":
    for s in ["@USER Sen tam bir 4pt4l mısın!!! URL", "s.a.l.a.k herif", "ÇOOOOK GÜZELLL bir maç",
              "IŞIK ve İSTANBUL", "#Galatasaray kazandııı"]:
        print(f"{s!r:45} -> {normalize(s)!r:35} | BERT: {light_normalize(s)!r}")
