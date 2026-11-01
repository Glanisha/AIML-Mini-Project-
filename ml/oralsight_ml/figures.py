"""Report/slide figures, drawn only from saved evaluation results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from .common import ARTIFACTS, FIGURES, load_json
from .data import load_manifest

TEAL, GREY, RED, AMBER = "#0f766e", "#94a3b8", "#dc2626", "#d97706"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def dataset_overview():
    df = load_manifest()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    counts = df.groupby(["split", "label"]).size().unstack()
    counts.plot.bar(ax=ax[0], color=[TEAL, RED], rot=0)
    ax[0].set_title("Images per split")
    ax[0].set_xlabel("")
    for label, c in [("benign", TEAL), ("malignant", RED)]:
        d = df[df.label == label]
        ax[1].scatter(d.width, d.height, s=8, alpha=0.6, color=c, label=label)
    ax[1].set_title("Original resolution (px)")
    ax[1].set_xlabel("width")
    ax[1].set_ylabel("height")
    ax[1].legend(frameon=False)
    sizes = df.group.value_counts().value_counts().sort_index()
    ax[2].bar(sizes.index.astype(str), sizes.values, color=GREY)
    ax[2].set_title("Near-duplicate group sizes")
    ax[2].set_xlabel("images per group")
    _save(fig, "dataset_overview.png")


def training_curves(full):
    archs = list(full["training"])
    fig, ax = plt.subplots(1, len(archs), figsize=(5 * len(archs), 3.2), squeeze=False)
    for a, axis in zip(archs, ax[0]):
        hist = load_json(ARTIFACTS / a / "history.json")["history"]
        for k, h in hist.items():
            e = [r["epoch"] for r in h]
            axis.plot(e, [r["train_loss"] for r in h], color=GREY, lw=0.8)
            axis.plot(e, [r["val_loss"] for r in h], color=TEAL, lw=0.8)
        axis.set_title(f"{a}: train (grey) / val (teal) loss per fold")
        axis.set_xlabel("epoch")
    _save(fig, "training_curves.png")


def roc_pr(full):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    h = full["headline"]
    r = np.array(full["curves"]["roc"])
    rb = np.array(full["curves"]["roc_baseline"])
    base_auc = next(c["roc_auc"] for c in full["comparison"] if c["key"] == "baseline")
    ax[0].plot(r[:, 0], r[:, 1], color=TEAL, lw=2, label=f"OralSight AUC {h['roc_auc']['value']:.3f} [{h['roc_auc']['ci95'][0]:.2f}–{h['roc_auc']['ci95'][1]:.2f}]")
    ax[0].plot(rb[:, 0], rb[:, 1], color=GREY, lw=1.5, ls="--", label=f"Baseline AUC {base_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], color="#e2e8f0", lw=1)
    m = full["headline"]
    ax[0].scatter([1 - m["specificity"]["value"]], [m["sensitivity"]["value"]], color=RED, zorder=5, label="Operating point")
    ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title="ROC (test set)")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    pr = np.array(full["curves"]["pr"])
    ax[1].plot(pr[:, 0], pr[:, 1], color=TEAL, lw=2, label=f"PR-AUC {h['pr_auc']['value']:.3f}")
    ax[1].axhline(full["test_class_counts"]["malignant"] / full["test_size"], color=GREY, ls="--", lw=1, label="Prevalence")
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision–recall (test set)", ylim=(0, 1.02))
    ax[1].legend(frameon=False, fontsize=7)
    _save(fig, "roc_pr.png")


def confusion(full):
    c = full["headline"]["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(mat, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=14, color="black")
    ax.set_xticks([0, 1], ["benign", "malignant"])
    ax.set_yticks([0, 1], ["benign", "malignant"])
    ax.set(xlabel="Predicted", ylabel="True", title=f"Test confusion @ threshold {full['operating_threshold']:.2f}")
    _save(fig, "confusion_matrix.png")


def reliability(full):
    cal = full["calibration"]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot([0, 1], [0, 1], color="#e2e8f0")
    for key, colour, label in [("reliability_before", GREY, f"Before (ECE {cal['ece_before']:.3f})"), ("reliability_after", TEAL, f"After temperature scaling (ECE {cal['ece_after']:.3f})")]:
        rows = cal[key]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows], "o-", color=colour, label=label, ms=4)
    ax.set(xlabel="Predicted P(malignant)", ylabel="Observed malignant rate", title="Reliability diagram (test set)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "reliability.png")


def risk_coverage(full):
    pts = full["risk_coverage"]["points"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot([p["coverage"] for p in pts], [p["risk"] for p in pts], color=TEAL, lw=2)
    sel = full["selective"]
    ax.axvline(sel["coverage"], color=AMBER, ls="--", lw=1, label=f"Deployed coverage {sel['coverage']:.0%}")
    ax.set(xlabel="Coverage (share of cases answered)", ylabel="Error rate on answered cases", title=f"Risk–coverage (AURC {full['risk_coverage']['aurc']:.3f})")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "risk_coverage.png")


def decision_curve(full):
    rows = full["decision_curve"]
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(t, [r["model"] for r in rows], color=TEAL, lw=2, label="OralSight")
    ax.plot(t, [r["treat_all"] for r in rows], color=GREY, ls="--", label="Refer all")
    ax.plot(t, [0 for _ in rows], color="black", lw=0.8, label="Refer none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve analysis (test set)", ylim=(-0.1, max(r["model"] for r in rows) + 0.05))
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "decision_curve.png")


def robustness(full):
    rob = full["robustness"]
    names = [k for k in rob if k != "clean"]
    fig, ax = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharey=True)
    for axis, name in zip(ax, names):
        levels = ["clean"] + [r["level"] for r in rob[name]]
        auc = [rob["clean"]["auc"]] + [r["auc"] for r in rob[name]]
        flag = [rob["clean"]["abstain_rate"]] + [r["abstain_rate"] for r in rob[name]]
        axis.plot(levels, auc, "o-", color=TEAL, label="AUC")
        axis.bar(levels, flag, color=AMBER, alpha=0.35, label="Abstain rate")
        axis.set_title(name.replace("_", " "))
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", labelsize=7)
    ax[0].legend(frameon=False, fontsize=7)
    _save(fig, "robustness.png")


def ood_histogram(full, hist):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.linspace(0, max(max(v) for v in hist.values()) * 1.02, 40)
    colours = [TEAL, RED, AMBER]
    for (name, v), c in zip(hist.items(), colours):
        ax.hist(v, bins=bins, alpha=0.55, color=c, label=name.replace("_", " "), density=True)
    ax.axvline(full["ood"]["threshold"], color="black", ls="--", lw=1, label="Threshold")
    ax.set(xlabel="Cosine distance to 5th-nearest training image", ylabel="density", title="Out-of-distribution scores")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "ood_scores.png")


def gradcam_gallery(test, images, model, case_rows):
    picks = []
    for outcome in ["TP", "TN", "FP", "FN", "abstained"]:
        idx = [i for i, r in enumerate(case_rows) if r["outcome"] == outcome][:2]
        picks += [(outcome, i) for i in idx]
    if not picks:
        return
    fig, ax = plt.subplots(2, len(picks), figsize=(2.1 * len(picks), 4.4))
    ax = np.array(ax).reshape(2, -1)
    for col, (outcome, i) in enumerate(picks):
        im = images[i].copy()
        im.thumbnail((512, 512))
        pred = model.predict(images[i])
        import base64
        import io

        from PIL import Image

        over = Image.open(io.BytesIO(base64.b64decode(pred.heatmap_jpeg_base64)))
        ax[0, col].imshow(im)
        ax[1, col].imshow(over)
        ax[0, col].set_title(f"{outcome} · true {case_rows[i]['label']}\nP(mal)={case_rows[i]['p_malignant']:.2f}", fontsize=7)
        for a in ax[:, col]:
            a.axis("off")
    _save(fig, "gradcam_gallery.png")


def make_all(full, ood_hist, test, images, model, case_rows):
    dataset_overview()
    training_curves(full)
    roc_pr(full)
    confusion(full)
    reliability(full)
    risk_coverage(full)
    decision_curve(full)
    robustness(full)
    ood_histogram(full, ood_hist)
    gradcam_gallery(test, images, model, case_rows)
