"""Area B - regex normalisation of noisy Turkish chat text.

Steps (applied in order):
  1. remove user mentions / URLs (OffensEval already replaces them with @USER / URL)
  2. Turkish-aware lower-casing  (I -> ı, İ -> i)
  3. leetspeak -> letters inside words  (s1kt1r -> siktir, @ -> a, $ -> s)
  4. collapse letters repeated 3+ times  (ooooç -> oç)
  5. fold Turkish characters to ASCII for the lexicon / tf-idf / word2vec models
     (BERTurk gets the un-folded text because its vocabulary knows ç, ğ, ı, ö, ş, ü)
"""
import re

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\bURL\b")
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
# a leet character counts only when it sits between letters, e.g. "s1kt1r", "m@l" (not "2020")
LEET_RE = re.compile(r"(?:(?<=[a-zçğıöşü])|(?<=\s)|^)[013457@$!](?=[a-zçğıöşü])")
REPEAT_RE = re.compile(r"(\w)\1{2,}")
NON_WORD_RE = re.compile(r"[^\w\s]")
SPACE_RE = re.compile(r"\s+")
TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str, fold: bool = True) -> str:
    t = MENTION_RE.sub(" ", text)
    t = URL_RE.sub(" ", t)
    t = tr_lower(t)
    t = LEET_RE.sub(lambda m: LEET_MAP[m.group(0)], t)
    t = REPEAT_RE.sub(r"\1", t)
    t = NON_WORD_RE.sub(" ", t)
    if fold:
        t = t.translate(TR_FOLD)
    return SPACE_RE.sub(" ", t).strip()


def tokenize(text: str):
    """Normalised, ASCII-folded word tokens (used by lexicon, tf-idf, word2vec)."""
    return TOKEN_RE.findall(normalize(text, fold=True))


def normalize_for_bert(text: str) -> str:
    """Light cleaning for BERTurk: keep case and Turkish letters, fix leet + repeats."""
    t = MENTION_RE.sub("@USER", text)
    t = URL_RE.sub("URL", t)
    t = re.sub(r"(?:(?<=[A-Za-zÇĞİÖŞÜçğıöşü])|(?<=\s)|^)[013457$](?=[A-Za-zÇĞİÖŞÜçğıöşü])",
               lambda m: LEET_MAP[m.group(0)], t)
    t = REPEAT_RE.sub(r"\1\1", t)
    return SPACE_RE.sub(" ", t).strip()


if __name__ == "__main__":
    for s in ["Bu adam tam bir s1kt1r0lup gitsin!!!", "Harikaaaaa bir maçtı İstanbul",
              "@USER $erefsiz herif URL"]:
        print(repr(s), "->", tokenize(s), "|", normalize_for_bert(s))
