"""R3 + R5: frozen-embedding baseline, OOD reference statistics, calibration and thresholds.

Everything here is fitted on the development set only (out-of-fold where needed).
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.inference import quality

from .common import ARTIFACTS, N_FOLDS, SEED, save_json
from .data import eval_tensor, load_image, load_manifest
from .models import FrozenEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OOD_K = 5  # k-th nearest neighbour (Sun et al., ICML 2022, "Out-of-distribution detection with deep nearest neighbors")
TARGET_SENSITIVITY = 0.90  # screening: prioritise not missing malignancy
TARGET_COVERAGE = 0.85  # abstain on roughly the least certain 15%
MIN_MEMBER_AGREEMENT = 0.6


@torch.no_grad()
def embed(paths: list[str]) -> np.ndarray:
    enc = FrozenEncoder().to(DEVICE).eval()
    x = eval_tensor(paths)
    return torch.cat([enc(x[i : i + 64].to(DEVICE)).cpu() for i in range(0, len(x), 64)]).numpy()


def _normalise(e: np.ndarray) -> np.ndarray:
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def fit_ood(emb: np.ndarray) -> dict:
    """Reference bank of L2-normalised training embeddings (stored as float16 to keep the bundle small)."""
    return {"bank": _normalise(emb).astype(np.float16), "k": np.array(OOD_K)}


def ood_distance(stats: dict, emb: np.ndarray) -> np.ndarray:
    """Cosine distance to the k-th nearest training image."""
    d = 1 - _normalise(emb) @ stats["bank"].astype(np.float32).T
    return np.sort(d, axis=1)[:, int(stats["k"]) - 1]


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    t_logits, t_y = torch.from_numpy(logits).double(), torch.from_numpy(y)
    grid = np.exp(np.linspace(np.log(0.25), np.log(6), 400))
    nll = [float(torch.nn.functional.cross_entropy(t_logits / t, t_y)) for t in grid]
    return float(grid[int(np.argmin(nll))])


def softmax_pos(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits) / t, 1)[:, 1].numpy()


def operating_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity is still >= target (maximises specificity)."""
    candidates = np.unique(p)[::-1]
    for t in candidates:
        if ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target:
            return float(t)
    return float(candidates[-1])


def quality_thresholds(paths: list[str]) -> tuple[dict, dict]:
    m = [quality.measure(load_image(p)) for p in paths]
    s = np.array([q.sharpness for q in m])
    b = np.array([q.brightness for q in m])
    bf = np.array([q.bright_fraction for q in m])
    df = np.array([q.dark_fraction for q in m])
    nz = np.array([q.noise for q in m])
    q = lambda a, pct: float(np.percentile(a, pct))
    thresholds = {
        "sharpness_warn": q(s, 5),
        "sharpness_reject": q(s, 1) * 0.5,
        "brightness_warn_low": q(b, 2.5),
        "brightness_reject_low": q(b, 1) * 0.6,
        "brightness_warn_high": q(b, 97.5),
        "brightness_reject_high": min(250.0, q(b, 99) + 0.5 * (255 - q(b, 99))),
        "bright_fraction_warn": max(q(bf, 97.5), 0.05),
        "dark_fraction_warn": max(q(df, 97.5), 0.2),
        "noise_warn": q(nz, 97.5),
        "noise_reject": q(nz, 99) * 1.5,
    }
    distribution = {
        name: {"p1": q(a, 1), "p5": q(a, 5), "median": q(a, 50), "p95": q(a, 95), "p99": q(a, 99)}
        for name, a in [("sharpness", s), ("brightness", b), ("bright_fraction", bf), ("dark_fraction", df), ("noise", nz)]
    }
    return thresholds, distribution


def run(arch: str) -> dict:
    df = load_manifest()
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    y_dev = dev.y.to_numpy()

    # --- frozen-embedding baseline (R3) + OOD reference space ---
    emb_dev, emb_test = embed(dev.path.tolist()), embed(test.path.tolist())
    np.savez(ARTIFACTS / "embeddings.npz", dev=emb_dev, test=emb_test)
    baseline = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, scoring="neg_log_loss"))
    baseline.fit(emb_dev, y_dev)
    np.save(ARTIFACTS / "baseline_test_probs.npy", baseline.predict_proba(emb_test)[:, 1])

    # OOD threshold: distances of held-out folds under stats fitted on the remaining folds
    held_out = np.zeros(len(dev))
    for k in range(N_FOLDS):
        tr, va = dev.fold != k, dev.fold == k
        held_out[va] = ood_distance(fit_ood(emb_dev[tr]), emb_dev[va])
    ood_stats = fit_ood(emb_dev)
    ood_threshold = float(np.percentile(held_out, 99))
    np.savez(ARTIFACTS / "ood_stats.npz", **ood_stats)

    # --- calibration + decision thresholds from out-of-fold logits ---
    oof = np.load(ARTIFACTS / arch / "oof.npz", allow_pickle=True)
    temperature = fit_temperature(oof["logits"], oof["y"])
    p_oof = softmax_pos(oof["logits"], temperature)
    op = operating_threshold(p_oof, oof["y"], TARGET_SENSITIVITY)
    margin = float(np.quantile(np.abs(p_oof - op), 1 - TARGET_COVERAGE))

    q_thresholds, q_distribution = quality_thresholds(dev.path.tolist())

    result = {
        "arch": arch,
        "temperature": temperature,
        "operating_threshold": op,
        "abstain_margin": margin,
        "min_member_agreement": MIN_MEMBER_AGREEMENT,
        "target_sensitivity": TARGET_SENSITIVITY,
        "target_coverage": TARGET_COVERAGE,
        "oof": {
            "auc": float(roc_auc_score(oof["y"], p_oof)),
            "sensitivity_at_op": float(((p_oof >= op) & (oof["y"] == 1)).sum() / (oof["y"] == 1).sum()),
            "specificity_at_op": float(((p_oof < op) & (oof["y"] == 0)).sum() / (oof["y"] == 0).sum()),
        },
        "ood": {"threshold": ood_threshold, "k": OOD_K, "held_out_distance_median": float(np.median(held_out))},
        "baseline": {"C": float(baseline[-1].C_[0])},
        "quality": q_thresholds,
        "quality_distribution": q_distribution,
    }
    save_json(result, ARTIFACTS / "calibration.json")
    print(f"T={temperature:.3f} op={op:.3f} margin={margin:.3f} ood_thr={ood_threshold:.2f} oof_auc={result['oof']['auc']:.3f}")
    return result
