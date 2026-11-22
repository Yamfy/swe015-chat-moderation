"""Area B: rule baseline (lexicon + edit distance) and tf-idf + logistic regression baseline."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config
from src.data import load_splits
from src.lexicon import LexiconMatcher, build_lexicon
from src.normalize import normalize

train, val, _ = load_splits()

# 1) lexicon learned from training data + edit-distance matcher
lex = LexiconMatcher(build_lexicon(train.text.tolist(), train.label.tolist()))
lex.save(config.ART_DIR / "lexicon.json")
val_pred = [int(lex.predict_proba(t) >= 0.5) for t in val.text]
print(f"Lexicon: {len(lex.lexicon)} words | top 15: {list(lex.lexicon)[:15]}")
print(f"Lexicon  val macro-F1 = {f1_score(val.label, val_pred, average='macro'):.4f}")

# 2) tf-idf (word 1-2 grams + char 2-5 grams, robust to misspellings) + logistic regression
pipe = Pipeline([
    ("tfidf", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)),
    ])),
    ("lr", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
])
pipe.fit([normalize(t) for t in train.text], train.label)
joblib.dump(pipe, config.ART_DIR / "tfidf_lr.joblib", compress=3)
val_pred = pipe.predict([normalize(t) for t in val.text])
print(f"tf-idf LR val macro-F1 = {f1_score(val.label, val_pred, average='macro'):.4f}")
