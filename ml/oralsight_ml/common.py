"""Paths, constants and small helpers shared by every pipeline stage."""

import json
import random
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent
DATA_RAW = ML_DIR / "data" / "raw"
DATA_EXTERNAL = ML_DIR / "data" / "external"
ARTIFACTS = ML_DIR / "artifacts"  # checkpoints, OOF predictions (git-ignored)
REPORTS = ML_DIR / "reports"  # metrics, figures, audit (committed)
FIGURES = REPORTS / "figures"
BUNDLE_DIR = REPO_DIR / "backend" / "app" / "model_bundle"
BACKEND_DATA = REPO_DIR / "backend" / "app" / "data"
SAMPLES_DIR = REPO_DIR / "backend" / "app" / "samples"

CLASSES = ["benign", "malignant"]
POSITIVE_CLASS = "malignant"
SEED = 42
N_FOLDS = 5

DATASET_CITATION = (
    "Chandrashekar H S, Geetha Kiran A, Murali S, Dinesh M S, Nanditha B R (2021). "
    "Oral Images Dataset, Mendeley Data, V2. doi:10.17632/mhjyrn35p4.2. Licensed CC BY 4.0."
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
