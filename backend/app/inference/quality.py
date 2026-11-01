"""Image quality gate: sharpness and exposure checks with data-driven thresholds.

Thresholds are derived from the training-image distribution by the ML pipeline
and stored in the model bundle config, so "blurry" means "blurrier than almost
every image the model learned from".
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

ANALYSIS_SIZE = 256


@dataclass
class QualityMetrics:
    sharpness: float  # variance of the Laplacian on a normalised grayscale image
    brightness: float  # mean luminance, 0..255
    dark_fraction: float  # share of near-black pixels
    bright_fraction: float  # share of blown-out pixels
    noise: float  # estimated sensor-noise sigma (Immerkaer, 1996)

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def measure(image: Image.Image) -> QualityMetrics:
    gray = np.asarray(image.convert("L").resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BILINEAR), dtype=np.float32)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    # Immerkaer's mask is the difference of two Laplacians, which cancels most image structure
    # and leaves noise; noise would otherwise inflate the sharpness score and pass the blur check
    m = (
        gray[:-2, :-2] - 2 * gray[:-2, 1:-1] + gray[:-2, 2:]
        - 2 * gray[1:-1, :-2] + 4 * gray[1:-1, 1:-1] - 2 * gray[1:-1, 2:]
        + gray[2:, :-2] - 2 * gray[2:, 1:-1] + gray[2:, 2:]
    )
    noise = float(np.sqrt(np.pi / 2) * np.abs(m).mean() / 6)
    return QualityMetrics(
        sharpness=float(lap.var()),
        brightness=float(gray.mean()),
        dark_fraction=float((gray < 20).mean()),
        bright_fraction=float((gray > 245).mean()),
        noise=noise,
    )


def assess(m: QualityMetrics, t: dict) -> tuple[str, list[str]]:
    """Return ("ok" | "warning" | "reject", reasons) using bundle thresholds `t`."""
    rejects: list[str] = []
    warnings: list[str] = []

    if m.sharpness < t["sharpness_reject"]:
        rejects.append("The image is very blurry. Hold the camera steady and tap to focus.")
    elif m.sharpness < t["sharpness_warn"]:
        warnings.append("The image looks slightly blurry, which can reduce accuracy.")

    if m.brightness < t["brightness_reject_low"]:
        rejects.append("The image is far too dark. Use more light.")
    elif m.brightness < t["brightness_warn_low"]:
        warnings.append("The image is darker than typical; more even lighting may help.")
    elif m.brightness > t["brightness_reject_high"]:
        rejects.append("The image is heavily overexposed. Avoid direct flash or strong light.")
    elif m.brightness > t["brightness_warn_high"]:
        warnings.append("The image is brighter than typical; reduce glare if possible.")

    if m.noise > t["noise_reject"]:
        rejects.append("The image is very grainy or noisy. Use more light so the camera needs less amplification.")
    elif m.noise > t["noise_warn"]:
        warnings.append("The image is grainier than typical, which can reduce accuracy.")

    if m.bright_fraction > t["bright_fraction_warn"]:
        warnings.append("Large glare or reflection areas detected.")
    if m.dark_fraction > t["dark_fraction_warn"]:
        warnings.append("Large areas of the image are in shadow.")

    if rejects:
        return "reject", rejects + warnings
    if warnings:
        return "warning", warnings
    return "ok", []
