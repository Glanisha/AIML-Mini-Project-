"""Upload validation: decode safely, fix orientation, reject anything unusable."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_SIDE_PX = 64
MAX_PIXELS = 40_000_000  # guards against decompression bombs
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImageError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def load_image(data: bytes) -> Image.Image:
    """Return an RGB image with EXIF orientation applied, or raise InvalidImageError."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image is too large. The maximum size is 10 MB.", status_code=413)

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()  # detects truncated / corrupted files
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        raise InvalidImageError("Could not read the image. It may be corrupted or not an image.", status_code=415)

    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError("Unsupported format. Use JPEG, PNG or WebP.", status_code=415)

    # Phone cameras store rotation in EXIF; apply it so the model sees the upright image
    image = ImageOps.exif_transpose(image).convert("RGB")

    if min(image.size) < MIN_SIDE_PX:
        raise InvalidImageError(f"Image is too small. Each side must be at least {MIN_SIDE_PX} px.")
    return image
