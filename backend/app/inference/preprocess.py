"""Model input preprocessing. Must match the evaluation transform used in training."""

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_for_model(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    # Whole-image resize (no centre crop) so off-centre lesions are never cut out
    return image.convert("RGB").resize((size, size), Image.BILINEAR)


def to_model_input(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB PIL image -> float32 array of shape (1, 3, size, size), ImageNet-normalised."""
    arr = np.asarray(resize_for_model(image, size), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None].copy()
