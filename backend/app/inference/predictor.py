"""OralSight inference: quality gate -> OOD check -> calibrated ensemble -> triage -> CAM.

Bundle layout (written by ml/oralsight_ml/export.py):
  config.json         classes, calibration, thresholds, quality + OOD settings
  member_<i>.onnx     fine-tuned CNN: input (B,3,224,224) -> logits (B,C), cam (B,C,h,w)
  ood_backbone.onnx   frozen ImageNet encoder: input -> embedding (B,D)
  ood.npz             L2-normalised training embeddings for k-NN out-of-distribution scoring
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import heatmap, quality
from .preprocess import to_model_input


@dataclass
class Uncertainty:
    confidence: float  # calibrated probability of the predicted class
    member_agreement: float  # share of ensemble members voting for the final label
    member_std: float  # spread of the members' positive-class probabilities
    score_range: tuple[float, float]  # min/max positive-class probability across members


@dataclass
class OODResult:
    in_distribution: bool
    score: float
    threshold: float


@dataclass
class Triage:
    level: str  # "refer" | "routine" | "indeterminate"
    title: str
    message: str


@dataclass
class Prediction:
    predicted_label: str | None
    class_scores: dict[str, float]
    positive_probability: float
    operating_threshold: float
    abstain_margin: float
    abstained: bool
    abstain_reasons: list[str]
    quality_status: str
    quality_reasons: list[str]
    quality_metrics: dict[str, float]
    uncertainty: Uncertainty
    ood: OODResult
    triage: Triage
    heatmap_jpeg_base64: str | None
    member_probabilities: list[float]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class OralSightModel:
    def __init__(self, bundle_dir: Path, threads: int | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.config = json.loads((self.bundle_dir / "config.json").read_text(encoding="utf-8"))
        self.version: str = self.config["version"]
        self.classes: list[str] = self.config["classes"]
        self.positive_index = self.classes.index(self.config["positive_class"])

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self.members = [
            ort.InferenceSession(str(self.bundle_dir / name), opts, providers=providers)
            for name in self.config["members"]
        ]
        self.ood_session = ort.InferenceSession(str(self.bundle_dir / self.config["ood"]["backbone"]), opts, providers=providers)
        stats = np.load(self.bundle_dir / self.config["ood"]["stats"])
        self.ood_bank = stats["bank"].astype(np.float32)
        self.ood_k = int(stats["k"])

    # --- building blocks (also used by the ML evaluation scripts) ---

    def ood_score(self, x: np.ndarray) -> float:
        """Cosine distance from the image to its k-th nearest training image (deep k-NN OOD)."""
        emb = self.ood_session.run(None, {"input": x})[0][0]
        emb = emb / np.linalg.norm(emb)
        return float(np.sort(1 - self.ood_bank @ emb)[self.ood_k - 1])

    def member_outputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated per-member probabilities (M, C) and CAMs (M, C, h, w)."""
        temperature = self.config["calibration"]["temperature"]
        probs, cams = [], []
        for sess in self.members:
            logits, cam = sess.run(None, {"input": x})
            probs.append(_softmax(logits[0] / temperature))
            cams.append(cam[0])
        return np.stack(probs), np.stack(cams)

    # --- full pipeline ---

    def predict(self, image: Image.Image, explain: bool = True) -> Prediction:
        cfg = self.config
        t0 = time.perf_counter()

        q_metrics = quality.measure(image)
        q_status, q_reasons = quality.assess(q_metrics, cfg["quality"])
        x = to_model_input(image, cfg["input"]["size"])
        t1 = time.perf_counter()

        ood_threshold = cfg["ood"]["threshold"]
        ood_score = self.ood_score(x)
        member_probs, cams = self.member_outputs(x)
        t2 = time.perf_counter()

        mean_probs = member_probs.mean(axis=0)
        p_pos = float(mean_probs[self.positive_index])
        op = cfg["decision"]["operating_threshold"]
        band = cfg["decision"]["abstain_margin"]

        label_idx = self.positive_index if p_pos >= op else 1 - self.positive_index
        member_pos = member_probs[:, self.positive_index]
        votes = np.where(member_pos >= op, self.positive_index, 1 - self.positive_index)

        uncertainty = Uncertainty(
            confidence=round(float(mean_probs[label_idx]), 4),
            member_agreement=round(float((votes == label_idx).mean()), 4),
            member_std=round(float(member_pos.std()), 4),
            score_range=(round(float(member_pos.min()), 4), round(float(member_pos.max()), 4)),
        )
        ood = OODResult(in_distribution=ood_score <= ood_threshold, score=round(ood_score, 4), threshold=round(ood_threshold, 4))

        reasons: list[str] = []
        if q_status == "reject":
            reasons.append("Image quality is too low for a reliable result.")
        if not ood.in_distribution:
            reasons.append("This image does not resemble the oral lesion photos the model was trained on.")
        if abs(p_pos - op) < band:
            reasons.append("The model's score is too close to its decision threshold.")
        if uncertainty.member_agreement < cfg["decision"]["min_member_agreement"]:
            reasons.append("The ensemble models disagree with each other.")
        abstained = bool(reasons)

        heatmap_b64 = None
        if explain:
            # Average each member's normalised map for the class being explained
            cam = np.mean([heatmap.normalise_cam(c[label_idx]) for c in cams], axis=0)
            heatmap_b64 = heatmap.to_jpeg_base64(heatmap.overlay(image, cam))
        t3 = time.perf_counter()

        return Prediction(
            predicted_label=None if abstained else self.classes[label_idx],
            class_scores={c: round(float(p), 4) for c, p in zip(self.classes, mean_probs)},
            positive_probability=round(p_pos, 4),
            operating_threshold=op,
            abstain_margin=round(band, 4),
            abstained=abstained,
            abstain_reasons=reasons,
            quality_status=q_status,
            quality_reasons=q_reasons,
            quality_metrics=q_metrics.as_dict(),
            uncertainty=uncertainty,
            ood=ood,
            triage=self._triage(abstained, label_idx, q_status, ood.in_distribution),
            heatmap_jpeg_base64=heatmap_b64,
            member_probabilities=[round(float(p), 4) for p in member_pos],
            timings_ms={
                "preprocess": round((t1 - t0) * 1000, 1),
                "inference": round((t2 - t1) * 1000, 1),
                "explain": round((t3 - t2) * 1000, 1),
            },
        )

    def _triage(self, abstained: bool, label_idx: int, q_status: str, in_distribution: bool) -> Triage:
        # Abstentions get advice that matches the cause: fix the photo, or get the lesion seen
        if abstained and not in_distribution:
            return Triage(
                "indeterminate",
                "Photo not recognised",
                "This does not look like a close-up photo of a lesion inside the mouth. Take a clear, well-lit, close photo of the area of concern.",
            )
        if abstained and q_status == "reject":
            return Triage(
                "indeterminate",
                "Photo quality too low",
                "Retake the photo with steady hands and bright, even light. If you cannot get a clear photo, have the lesion examined by a dentist or doctor.",
            )
        if abstained:
            return Triage(
                "indeterminate",
                "Uncertain: get it checked",
                "The model cannot confidently classify this lesion. Uncertain cases should be examined by a dentist or doctor rather than ignored.",
            )
        if label_idx == self.positive_index:
            return Triage(
                "refer",
                "Higher suspicion: seek prompt examination",
                "This image resembles malignant lesions in the training data. Arrange an examination by a "
                "dentist or oral medicine specialist soon. Only a clinical examination and biopsy can confirm a diagnosis.",
            )
        return Triage(
            "routine",
            "Lower suspicion",
            "This image resembles benign lesions in the training data. This does not rule out disease: any mouth "
            "ulcer, patch or lump that lasts longer than 2 weeks should be examined by a clinician.",
        )
