"""Area B - rule baseline: offensive-word lexicon + minimum edit distance matching.

The lexicon is learned from the TRAINING split only (no external word list):
words that appear >= MIN_COUNT times and whose smoothed log-odds of appearing in an
offensive message is above a threshold. A message is flagged if any of its tokens
is within edit distance <= 1 (for tokens of length >= 5, so misspellings like
"serefsz" still match) of a lexicon entry.
"""
import json
import math
from collections import Counter

from .normalize import tokenize

MIN_COUNT = 5
LOG_ODDS_THRESHOLD = 1.5
MAX_LEXICON = 400


def min_edit_distance(a: str, b: str, max_d: int = 99) -> int:
    """Levenshtein distance with insert = delete = substitute = 1 (dynamic programming)."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


def build_lexicon(texts, labels):
    off, clean = Counter(), Counter()
    for t, y in zip(texts, labels):
        (off if y == 1 else clean).update(set(tokenize(t)))
    n_off, n_clean = sum(1 for y in labels if y == 1), sum(1 for y in labels if y == 0)
    scored = []
    for w in set(off) | set(clean):
        if off[w] + clean[w] < MIN_COUNT or len(w) < 3:
            continue
        # add-one smoothed document frequencies -> log-odds ratio
        lo = math.log((off[w] + 1) / (n_off + 2)) - math.log((clean[w] + 1) / (n_clean + 2))
        if lo >= LOG_ODDS_THRESHOLD:
            scored.append((lo, w))
    scored.sort(reverse=True)
    return {w: round(lo, 3) for lo, w in scored[:MAX_LEXICON]}


class LexiconMatcher:
    name = "Lexicon + edit distance (rules)"

    def __init__(self, lexicon: dict):
        self.lexicon = lexicon
        self.by_len = {}
        for w in lexicon:
            self.by_len.setdefault(len(w), []).append(w)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf-8")))

    def save(self, path):
        json.dump(self.lexicon, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def match(self, token: str):
        if token in self.lexicon:
            return token
        if len(token) < 5:
            return None
        for L in (len(token) - 1, len(token), len(token) + 1):
            for w in self.by_len.get(L, []):
                if min_edit_distance(token, w, max_d=1) <= 1:
                    return w
        return None

    def predict_proba(self, text: str) -> float:
        """Returns P(offensive): squashed sum of matched log-odds (0 if no match)."""
        score = 0.0
        for tok in tokenize(text):
            hit = self.match(tok)
            if hit:
                score += self.lexicon[hit]
        return 0.0 if score == 0 else 1 - math.exp(-score)  # 1 hit of lo=1.5 -> 0.78
