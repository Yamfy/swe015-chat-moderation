"""Download and split OffensEval-TR 2020 (Coltekin, 2020).

Splits:
  train / val : stratified 90/10 split of the official training file (seed 42)
  test        : the official OffensEval-2020 Turkish test set (held out, used by ALL models)
"""
import csv
import io
import sys
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

csv.field_size_limit(sys.maxsize)
LABEL_MAP = {"NOT": 0, "OFF": 1}


def _zip_bytes() -> bytes:
    local = config.DATA_DIR / "offenseval2020-turkish.zip"
    if config.DATA_ZIP:
        return open(config.DATA_ZIP, "rb").read()
    if not local.exists():
        print(f"Downloading {config.DATA_URL} ...")
        urllib.request.urlretrieve(config.DATA_URL, local)
    return local.read_bytes()


def _find(zf: zipfile.ZipFile, suffix: str) -> str:
    for name in zf.namelist():
        if name.endswith(suffix):
            return name
    raise FileNotFoundError(suffix)


def _read_tsv(zf, name):
    text = zf.read(name).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))


def load_raw():
    zf = zipfile.ZipFile(io.BytesIO(_zip_bytes()))
    train_rows = _read_tsv(zf, _find(zf, "offenseval-tr-training-v1.tsv"))
    test_rows = _read_tsv(zf, _find(zf, "offenseval-tr-testset-v1.tsv"))
    # label file: comma separated "id,label", no header, aligned with the test file by position
    lab_text = zf.read(_find(zf, "offenseval-tr-labela-v1.tsv")).decode("utf-8")
    test_labels = [line.split(",")[1].strip() for line in lab_text.splitlines() if line.strip()]
    assert len(test_labels) == len(test_rows), "test labels do not align with test rows"

    train = pd.DataFrame({"text": [r["tweet"] for r in train_rows],
                          "label": [LABEL_MAP[r["subtask_a"].strip()] for r in train_rows]})
    test = pd.DataFrame({"text": [r["tweet"] for r in test_rows],
                         "label": [LABEL_MAP[l] for l in test_labels]})
    return train, test


def load_splits():
    """Returns (train, val, test) DataFrames with columns text, label. Cached as CSV."""
    paths = [config.DATA_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if all(p.exists() for p in paths):
        return tuple(pd.read_csv(p, keep_default_na=False) for p in paths)
    full_train, test = load_raw()
    train, val = train_test_split(full_train, test_size=config.VAL_FRACTION,
                                  stratify=full_train["label"], random_state=config.SEED)
    for df, p in zip((train, val, test), paths):
        df.reset_index(drop=True).to_csv(p, index=False)
    return tuple(pd.read_csv(p, keep_default_na=False) for p in paths)


if __name__ == "__main__":
    tr, va, te = load_splits()
    for n, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{n:5s}: {len(d):6d} messages, {d.label.mean():.1%} offensive")
