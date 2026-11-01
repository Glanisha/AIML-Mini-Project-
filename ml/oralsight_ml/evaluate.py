"""R7: evaluate the locked model bundle on the untouched test set.

All headline numbers come from `app.inference.predictor.OralSightModel`, the same
code the API serves. Writes ml/reports/metrics_full.json, figures, and the
app-facing backend/app/data/metrics.json plus the demo sample pack.
"""

import io
import shutil
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from app.inference.predictor import OralSightModel, _softmax
from app.inference.preprocess import to_model_input

from .common import (
    ARTIFACTS,
    BACKEND_DATA,
    BUNDLE_DIR,
    DATA_EXTERNAL,
    DATASET_CITATION,
    REPORTS,
    SAMPLES_DIR,
    SEED,
    load_json,
    save_json,
)
from .data import load_image, load_manifest

N_BOOT = 2000
RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------- metrics

def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": (sens + spec) / 2,
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error on the positive-class probability, with reliability bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            total += m.mean() * abs(conf - acc)
            rows.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "mean_predicted": conf, "observed_rate": acc, "count": int(m.sum())})
    return float(total), rows


def bootstrap(y: np.ndarray, p: np.ndarray, fn, n: int = N_BOOT) -> list[float]:
    """Stratified percentile bootstrap 95% CI."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        v = fn(y[i], p[i])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def with_ci(y, p, thr) -> dict:
    point = binary_metrics(y, p, thr)
    out = {"roc_auc": {"value": float(roc_auc_score(y, p)), "ci95": bootstrap(y, p, roc_auc_score)},
           "pr_auc": {"value": float(average_precision_score(y, p)), "ci95": bootstrap(y, p, average_precision_score)}}
    for key in ["sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1"]:
        out[key] = {"value": point[key], "ci95": bootstrap(y, p, lambda a, b, k=key: binary_metrics(a, b, thr)[k])}
    out["confusion"] = {k: point[k] for k in ["tp", "tn", "fp", "fn"]}
    return out


def paired_auc_diff(y, p_a, p_b) -> dict:
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = []
    for _ in range(N_BOOT):
        i = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    diffs = np.array(diffs)
    return {"difference": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_value_one_sided": float((diffs <= 0).mean())}


def curve_points(x, y, max_points: int = 60) -> list[list[float]]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), max_points)).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def risk_coverage(y, p, thr) -> dict:
    order = np.argsort(-np.abs(p - thr))  # most certain first
    correct = ((p >= thr).astype(int) == y)[order]
    pos = (y == 1)[order]
    pred_pos = (p >= thr)[order]
    pts = []
    for k in range(1, len(y) + 1):
        tp = (pos[:k] & pred_pos[:k]).sum()
        sens = tp / pos[:k].sum() if pos[:k].sum() else float("nan")
        pts.append({"coverage": k / len(y), "risk": 1 - correct[:k].mean(), "sensitivity": float(sens)})
    aurc = float(np.mean([pt["risk"] for pt in pts]))
    return {"aurc": aurc, "points": pts[:: max(1, len(pts) // 50)] + [pts[-1]]}


def decision_curve(y, p) -> list[dict]:
    prev, n = y.mean(), len(y)
    rows = []
    for t in np.round(np.arange(0.05, 0.81, 0.025), 3):
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        w = t / (1 - t)
        rows.append({"threshold": float(t), "model": float(tp / n - fp / n * w), "treat_all": float(prev - (1 - prev) * w), "treat_none": 0.0})
    return rows


# ---------------------------------------------------------------- pipeline helpers

def run_pipeline(model: OralSightModel, images: list[Image.Image]) -> dict:
    preds = [model.predict(im, explain=False) for im in images]
    return {
        "p": np.array([r.positive_probability for r in preds]),
        "abstained": np.array([r.abstained for r in preds]),
        "ood_score": np.array([r.ood.score for r in preds]),
        "quality": np.array([r.quality_status for r in preds]),
        "reasons": [r.abstain_reasons for r in preds],
        "member_std": np.array([r.uncertainty.member_std for r in preds]),
    }


def uncalibrated_probs(model: OralSightModel, images) -> np.ndarray:
    out = []
    for im in images:
        x = to_model_input(im)
        out.append(np.mean([_softmax(s.run(None, {"input": x})[0][0])[model.positive_index] for s in model.members]))
    return np.array(out)


def jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def noise(im: Image.Image, sigma: float) -> Image.Image:
    a = np.asarray(im, dtype=np.float32) + np.random.default_rng(SEED).normal(0, sigma, (im.height, im.width, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


CORRUPTIONS = {
    "gaussian_blur": [("σ=1", lambda im: im.filter(ImageFilter.GaussianBlur(1))), ("σ=3", lambda im: im.filter(ImageFilter.GaussianBlur(3))), ("σ=6", lambda im: im.filter(ImageFilter.GaussianBlur(6)))],
    "jpeg_compression": [("q=50", lambda im: jpeg(im, 50)), ("q=20", lambda im: jpeg(im, 20)), ("q=5", lambda im: jpeg(im, 5))],
    "darken": [("×0.7", lambda im: ImageEnhance.Brightness(im).enhance(0.7)), ("×0.45", lambda im: ImageEnhance.Brightness(im).enhance(0.45)), ("×0.2", lambda im: ImageEnhance.Brightness(im).enhance(0.2))],
    "brighten": [("×1.3", lambda im: ImageEnhance.Brightness(im).enhance(1.3)), ("×1.7", lambda im: ImageEnhance.Brightness(im).enhance(1.7)), ("×2.5", lambda im: ImageEnhance.Brightness(im).enhance(2.5))],
    "sensor_noise": [("σ=8", lambda im: noise(im, 8)), ("σ=20", lambda im: noise(im, 20)), ("σ=40", lambda im: noise(im, 40))],
}


def shrink(im: Image.Image, side: int = 1024) -> Image.Image:
    im = im.copy()
    im.thumbnail((side, side))
    return im


def ood_images() -> dict[str, list[Image.Image]]:
    import pandas as pd

    # Oxford Flowers-102 test split (Hugging Face parquet mirror), a far-OOD natural-photo proxy
    flowers = pd.read_parquet(DATA_EXTERNAL / "flowers_test.parquet")
    idx = np.random.default_rng(SEED).choice(len(flowers), 200, replace=False)
    natural = [Image.open(io.BytesIO(flowers.iloc[int(i)]["image"]["bytes"])).convert("RGB") for i in idx]
    rng = np.random.default_rng(SEED)
    synthetic = []
    for i in range(60):
        kind = i % 4
        if kind == 0:
            a = rng.integers(0, 256, (256, 256, 3))
        elif kind == 1:
            a = np.ones((256, 256, 3)) * rng.integers(0, 256, 3)
        elif kind == 2:
            g = np.linspace(0, 255, 256)
            a = np.stack([np.tile(g, (256, 1)), np.tile(g[:, None], (1, 256)), np.full((256, 256), rng.integers(0, 256))], -1)
        else:
            a = (np.indices((256, 256)).sum(0) // rng.integers(8, 48) % 2)[..., None] * rng.integers(0, 256, 3)
        synthetic.append(Image.fromarray(a.astype(np.uint8)))
    return {"natural_photos_flowers102": natural, "synthetic_patterns": synthetic}


# ---------------------------------------------------------------- main

def run(arch_names: list[str]) -> dict:
    from . import figures

    df = load_manifest()
    test = df[df.split == "test"].reset_index(drop=True)
    y = test.y.to_numpy()
    images = [load_image(p) for p in test.path]
    model = OralSightModel(BUNDLE_DIR)
    cfg = model.config
    op = cfg["decision"]["operating_threshold"]

    print("Running deployed pipeline on test set...")
    res = run_pipeline(model, images)
    p = res["p"]
    keep = ~res["abstained"]

    headline = with_ci(y, p, op)
    at_half = binary_metrics(y, p, 0.5)
    selective = binary_metrics(y[keep], p[keep], op) if keep.any() else {}

    p_uncal = uncalibrated_probs(model, images)
    ece_uncal, rel_uncal = ece(y, p_uncal)
    ece_cal, rel_cal = ece(y, p)

    # --- comparison table: baseline vs each fine-tuned ensemble (threshold-free AUC + acc@0.5) ---
    comparison = []
    p_base = np.load(ARTIFACTS / "baseline_test_probs.npy")
    comparison.append({"name": "Baseline: frozen ImageNet embeddings + logistic regression", "key": "baseline",
                       "roc_auc": float(roc_auc_score(y, p_base)), **binary_metrics(y, p_base, 0.5)})
    for arch in arch_names:
        tl = np.load(ARTIFACTS / arch / "test_logits.npy")
        pa = np.mean([_softmax(l)[:, 1] for l in tl], axis=0)
        single = [roc_auc_score(y, _softmax(l)[:, 1]) for l in tl]
        hist = load_json(ARTIFACTS / arch / "history.json")["summary"]
        comparison.append({"name": f"Fine-tuned {arch} (5-model ensemble)", "key": arch, "roc_auc": float(roc_auc_score(y, pa)),
                           "single_model_auc_mean": float(np.mean(single)), "oof_auc": hist["oof_auc"], **binary_metrics(y, pa, 0.5)})
    comparison.append({"name": f"Deployed OralSight pipeline (calibrated, threshold {op:.2f})", "key": "deployed",
                       "roc_auc": float(roc_auc_score(y, p)), **binary_metrics(y, p, op)})
    vs_baseline = paired_auc_diff(y, p, p_base)

    # --- robustness ---
    print("Robustness sweep...")
    small = [shrink(im) for im in images]
    robustness = {"clean": {"auc": float(roc_auc_score(y, p)), **_rob_summary(y, res, op)}}
    for name, levels in CORRUPTIONS.items():
        robustness[name] = []
        for label, fn in levels:
            r = run_pipeline(model, [fn(im) for im in small])
            robustness[name].append({"level": label, "auc": float(roc_auc_score(y, r["p"])), **_rob_summary(y, r, op)})

    # --- OOD detection ---
    print("OOD evaluation...")
    ood_sets = ood_images()
    in_scores = res["ood_score"]
    ood = {"threshold": cfg["ood"]["threshold"], "in_distribution_test_flagged_rate": float((in_scores > cfg["ood"]["threshold"]).mean()), "sets": {}}
    ood_hist = {"in_distribution_test": in_scores.tolist()}
    for name, ims in ood_sets.items():
        s = np.array([model.ood_score(to_model_input(im)) for im in ims])
        lab = np.r_[np.zeros(len(in_scores)), np.ones(len(s))]
        ood["sets"][name] = {"n": len(s), "auroc": float(roc_auc_score(lab, np.r_[in_scores, s])), "detected_rate": float((s > cfg["ood"]["threshold"]).mean())}
        ood_hist[name] = s.tolist()

    # --- latency (CPU, full pipeline incl. explanation) ---
    print("Latency benchmark...")
    latency = {}
    for threads in [1, 2, None]:
        m = OralSightModel(BUNDLE_DIR, threads=threads)
        im = small[0]
        m.predict(im)
        t = []
        for _ in range(20):
            t0 = time.perf_counter()
            m.predict(im)
            t.append((time.perf_counter() - t0) * 1000)
        latency[f"threads_{threads or 'all'}"] = {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}

    # --- error review + demo pack ---
    case_rows = []
    for i, row in test.iterrows():
        case_rows.append({"path": row.path, "label": row.label, "p_malignant": float(p[i]), "abstained": bool(res["abstained"][i]),
                          "reasons": res["reasons"][i], "quality": str(res["quality"][i]), "ood_score": float(res["ood_score"][i]),
                          "member_std": float(res["member_std"][i]),
                          "outcome": _outcome(y[i], p[i] >= op, res["abstained"][i])})
    save_json(case_rows, REPORTS / "test_case_review.json")
    samples = build_samples(model, test, images, p, res, op)

    roc = roc_curve(y, p)
    roc_base = roc_curve(y, p_base)
    pr = precision_recall_curve(y, p)
    full = {
        "model_version": cfg["version"],
        "architecture": cfg["architecture"],
        "dataset": {"citation": DATASET_CITATION, "audit": _audit_summary()},
        "test_size": int(len(y)),
        "test_class_counts": {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())},
        "operating_threshold": op,
        "decision_config": cfg["decision"],
        "temperature": cfg["calibration"]["temperature"],
        "headline": headline,
        "at_threshold_0_5": at_half,
        "selective": {"coverage": float(keep.mean()), "abstained": int((~keep).sum()), **selective},
        "abstain_reason_counts": _reason_counts(res["reasons"]),
        "calibration": {"ece_before": ece_uncal, "ece_after": ece_cal, "brier_before": float(brier_score_loss(y, p_uncal)),
                        "brier_after": float(brier_score_loss(y, p)), "reliability_before": rel_uncal, "reliability_after": rel_cal},
        "comparison": comparison,
        "deployed_vs_baseline_auc": vs_baseline,
        "curves": {"roc": curve_points(roc[0], roc[1]), "roc_baseline": curve_points(roc_base[0], roc_base[1]),
                   "pr": curve_points(pr[1][::-1], pr[0][::-1])},
        "risk_coverage": risk_coverage(y, p, op),
        "decision_curve": decision_curve(y, p),
        "robustness": robustness,
        "ood": ood,
        "latency_cpu": latency,
        "calibration_fit": load_json(ARTIFACTS / "calibration.json"),
        "training": {a: load_json(ARTIFACTS / a / "history.json")["summary"] for a in arch_names},
    }
    save_json(full, REPORTS / "metrics_full.json")
    save_json({"in": ood_hist}, ARTIFACTS / "ood_scores.json")
    figures.make_all(full, ood_hist, test, images, model, case_rows)
    save_json(app_metrics(full), BACKEND_DATA / "metrics.json")
    print(f"Test AUC {headline['roc_auc']['value']:.3f} {headline['roc_auc']['ci95']}, sens {headline['sensitivity']['value']:.3f}, "
          f"spec {headline['specificity']['value']:.3f}, coverage {keep.mean():.2f}, samples {len(samples)}")
    return full


def _rob_summary(y, r, op) -> dict:
    m = binary_metrics(y, r["p"], op)
    return {"sensitivity": m["sensitivity"], "specificity": m["specificity"], "accuracy": m["accuracy"],
            "abstain_rate": float(r["abstained"].mean()), "quality_reject_rate": float((r["quality"] == "reject").mean()),
            "quality_flag_rate": float((r["quality"] != "ok").mean())}


def _outcome(y, pred_pos, abstained) -> str:
    if abstained:
        return "abstained"
    return {(1, True): "TP", (0, False): "TN", (0, True): "FP", (1, False): "FN"}[(int(y), bool(pred_pos))]


def _reason_counts(reasons) -> dict:
    out = {}
    for rs in reasons:
        for r in rs:
            out[r] = out.get(r, 0) + 1
    return out


def _audit_summary() -> dict:
    a = load_json(REPORTS / "dataset_audit.json")
    keys = ["n_images", "class_counts", "patient_ids_available", "exact_duplicate_pairs", "near_duplicate_pairs", "n_groups", "split_counts", "group_leakage_across_splits"]
    return {k: a[k] for k in keys}


def build_samples(model, test, images, p, res, op) -> list[dict]:
    """Curated, clearly labelled held-out cases for the in-app demo (dataset is CC BY 4.0)."""
    if SAMPLES_DIR.exists():
        shutil.rmtree(SAMPLES_DIR)
    SAMPLES_DIR.mkdir(parents=True)
    y = test.y.to_numpy()
    outcomes = np.array([_outcome(y[i], p[i] >= op, res["abstained"][i]) for i in range(len(y))])
    margin = np.abs(p - op)

    def pick(mask, n, most_confident=True):
        idx = np.where(mask)[0]
        return list(idx[np.argsort(-margin[idx] if most_confident else margin[idx])][:n])

    chosen = [("malignant", "Malignant lesion: correctly flagged", i) for i in pick(outcomes == "TP", 2)]
    chosen += [("benign", "Benign lesion: correctly classed lower-suspicion", i) for i in pick(outcomes == "TN", 2)]
    chosen += [("uncertain", "Borderline case: model abstains", i) for i in pick(outcomes == "abstained", 1, most_confident=False)]
    chosen += [("misclassified", "Known error: model is wrong", i) for i in pick(np.isin(outcomes, ["FP", "FN"]), 1)]

    samples = []
    for category, title, i in chosen:
        sid = f"{category}-{len(samples) + 1}"
        im = shrink(images[i], 900)
        im.save(SAMPLES_DIR / f"{sid}.jpg", quality=90)
        samples.append({"id": sid, "category": category, "title": title, "true_label": test.label[i], "source_file": test.path[i]})

    # Degraded example: blur a confidently-correct benign case to show the quality gate
    tn = pick(outcomes == "TN", 3)
    if tn:
        i = tn[-1]
        sid = f"degraded-{len(samples) + 1}"
        shrink(images[i], 900).filter(ImageFilter.GaussianBlur(8)).save(SAMPLES_DIR / f"{sid}.jpg", quality=90)
        samples.append({"id": sid, "category": "degraded", "title": "Degraded photo: heavy blur triggers quality gate", "true_label": test.label[i], "source_file": test.path[i]})

    save_json({"attribution": DATASET_CITATION, "note": "Held-out test images never used in training or model selection.", "samples": samples},
              SAMPLES_DIR / "samples.json")
    return samples


def app_metrics(full: dict) -> dict:
    """Compact, chart-ready subset served to the mobile app's Evidence screen."""
    h = full["headline"]
    return {
        "is_mock": False,
        "model_version": full["model_version"],
        "architecture": full["architecture"],
        "dataset": full["dataset"]["citation"],
        "audit": full["dataset"]["audit"],
        "test_size": full["test_size"],
        "test_class_counts": full["test_class_counts"],
        "operating_threshold": full["operating_threshold"],
        "headline": {k: h[k] for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy", "f1"]},
        "confusion_matrix": {"labels": ["benign", "malignant"],
                             "matrix": [[h["confusion"]["tn"], h["confusion"]["fp"]], [h["confusion"]["fn"], h["confusion"]["tp"]]]},
        "selective": full["selective"],
        "calibration": {k: full["calibration"][k] for k in ["ece_before", "ece_after", "brier_before", "brier_after", "reliability_after", "reliability_before"]},
        "comparison": [{k: c[k] for k in ["name", "key", "roc_auc", "sensitivity", "specificity", "accuracy"]} for c in full["comparison"]],
        "deployed_vs_baseline_auc": full["deployed_vs_baseline_auc"],
        "curves": full["curves"],
        "risk_coverage": full["risk_coverage"],
        "decision_curve": full["decision_curve"],
        "robustness": full["robustness"],
        "ood": full["ood"],
        "cpu_latency_ms": full["latency_cpu"]["threads_2"]["p50_ms"],
        "latency_cpu": full["latency_cpu"],
        "notes": [
            f"Single-centre dataset of {full['dataset']['audit']['n_images']} images; no patient IDs, so near-duplicates were grouped as a proxy.",
            f"Test set has only {full['test_size']} images: confidence intervals are wide.",
            "Class balance (~50% malignant) is far above real screening prevalence; PPV in practice would be much lower.",
            "No demographic or clinical metadata, so subgroup performance could not be assessed.",
            "Scores are somewhat under-confident in the 30-70% range (temperature was fitted on single models, then 5 were averaged).",
            "Heatmaps show model attention, not verified lesion boundaries.",
            "Research prototype: not a medical device and not a diagnosis.",
        ],
    }
