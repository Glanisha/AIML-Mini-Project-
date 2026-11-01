"""Render class activation maps as a colour overlay on the original image."""

import base64
import io

import numpy as np
from PIL import Image

MAX_SIDE = 512


def _jet(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def normalise_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)  # keep evidence *for* the class, as in Grad-CAM
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """`cam` is a low-resolution (h, w) map; returns an RGB overlay image."""
    base = image.convert("RGB")
    base.thumbnail((MAX_SIDE, MAX_SIDE))
    cam_img = Image.fromarray((normalise_cam(cam) * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    heat = np.asarray(cam_img, dtype=np.float32) / 255.0
    colour = _jet(heat) * 255
    # Weight the overlay by activation so low-attention regions stay readable
    weight = (alpha * (0.35 + 0.65 * heat))[..., None]
    out = np.asarray(base, dtype=np.float32) * (1 - weight) + colour * weight
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def to_jpeg_base64(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
