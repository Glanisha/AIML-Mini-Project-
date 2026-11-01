"""Inference wrapper. Loaded once at startup (see main.lifespan).

MOCK until Ralph's model bundle is integrated (task P4). To integrate:
  - load weights, class order and calibration/abstention config in `load()`
  - apply Ralph's preprocessing + quality rules + model in `predict()`
  - keep the return type unchanged so the API contract stays stable
"""

from dataclasses import dataclass

from PIL import Image

from .schemas import QualityReport


@dataclass
class Prediction:
    predicted_label: str | None
    class_scores: dict[str, float]
    quality: QualityReport
    abstained: bool
    heatmap_png_base64: str | None


class Predictor:
    version = "mock-0.0.0"
    is_mock = True

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def predict(self, image: Image.Image) -> Prediction:
        return Prediction(
            predicted_label="benign",
            class_scores={"benign": 0.72, "malignant": 0.28},
            quality=QualityReport(status="ok", reasons=[]),
            abstained=False,
            heatmap_png_base64=None,
        )
