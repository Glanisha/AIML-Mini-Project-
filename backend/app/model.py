"""Model loading for the API. Uses the real ONNX bundle when present, else a labelled mock."""

import os
from pathlib import Path

from PIL import Image

from .inference.predictor import OODResult, Prediction, Triage, Uncertainty

BUNDLE_DIR = Path(os.getenv("MODEL_BUNDLE_DIR", Path(__file__).parent / "model_bundle"))


class MockModel:
    """Stand-in so the API and app work before a bundle is trained. Always flagged is_mock."""

    version = "mock-0.0.0"
    is_mock = True
    positive_class = "malignant"

    def predict(self, image: Image.Image, explain: bool = True) -> Prediction:
        return Prediction(
            predicted_label=None,
            class_scores={"benign": 0.5, "malignant": 0.5},
            positive_probability=0.5,
            operating_threshold=0.5,
            abstain_margin=0.0,
            abstained=True,
            abstain_reasons=["No trained model bundle is installed on the server."],
            quality_status="ok",
            quality_reasons=[],
            quality_metrics={},
            uncertainty=Uncertainty(0.5, 1.0, 0.0, (0.5, 0.5)),
            ood=OODResult(True, 0.0, 0.0),
            triage=Triage("indeterminate", "Demo mode", "The server is running without a trained model."),
            heatmap_jpeg_base64=None,
            member_probabilities=[],
        )


def load_model():
    if (BUNDLE_DIR / "config.json").exists():
        from .inference.predictor import OralSightModel

        model = OralSightModel(BUNDLE_DIR, threads=int(os.getenv("ORT_THREADS", "0")) or None)
        model.is_mock = False
        model.positive_class = model.config["positive_class"]
        return model
    return MockModel()
