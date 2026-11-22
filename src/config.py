"""Central configuration: paths, model names, hyper-parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"          # trained weights (NOT committed, see .gitignore)
RESULTS_DIR = ROOT / "results"        # metrics tables produced by evaluate.py

DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
# Optional: point to a local copy of the zip instead of downloading it
DATA_ZIP = os.environ.get("SWE015_DATA_ZIP")

# Hugging Face Hub repo that stores the trained weights (set in README / Colab)
HF_REPO_ID = os.environ.get("SWE015_HF_REPO", "")

BERT_NAME = os.environ.get("SWE015_BERT", "dbmdz/bert-base-turkish-cased")  # BERTurk
MAX_LEN = 64
SEED = 42
VAL_FRACTION = 0.1

LABELS = ["clean", "offensive"]   # 0 = NOT, 1 = OFF in OffensEval
LATENCY_BUDGET_MS = 50.0          # our CPU budget per message
LATENCY_RUNS = 200                # >= 100 runs required by the homework

for d in (DATA_DIR, ART_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
