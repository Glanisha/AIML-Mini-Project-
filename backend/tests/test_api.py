import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan, so the model is loaded
        yield c


def make_image(fmt="JPEG", size=(256, 256), exif_orientation=None) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (200, 120, 120))
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def post(client, data: bytes, name="photo.jpg", content_type="image/jpeg"):
    return client.post("/analyze", files={"file": (name, data, content_type)})


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_analyze_accepts_supported_formats(client, fmt):
    res = post(client, make_image(fmt))
    assert res.status_code == 200
    body = res.json()
    assert set(body) >= {"predicted_label", "class_scores", "quality", "abstained", "heatmap_available", "model_version", "latency_ms"}
    assert body["heatmap_available"] == (body["heatmap_png_base64"] is not None)


def test_analyze_ignores_client_content_type(client):
    # Mobile clients sometimes send application/octet-stream; validation is by decoding
    assert post(client, make_image(), content_type="application/octet-stream").status_code == 200


def test_analyze_handles_exif_rotation(client):
    assert post(client, make_image(size=(300, 200), exif_orientation=6)).status_code == 200


def test_rejects_non_image(client):
    res = post(client, b"definitely not an image", name="notes.txt", content_type="text/plain")
    assert res.status_code == 415


def test_rejects_corrupted_image(client):
    assert post(client, make_image()[:200]).status_code == 415


def test_rejects_unsupported_format(client):
    assert post(client, make_image("GIF"), name="a.gif", content_type="image/gif").status_code == 415


def test_rejects_empty_file(client):
    assert post(client, b"").status_code == 422


def test_rejects_tiny_image(client):
    assert post(client, make_image(size=(20, 20))).status_code == 422


def test_rejects_oversized_upload(client):
    assert post(client, b"0" * (10 * 1024 * 1024 + 10)).status_code == 413


def test_repeated_requests_are_stable(client):
    data = make_image()
    results = [post(client, data).json()["class_scores"] for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_metrics_served(client):
    body = client.get("/metrics").json()
    assert "models" in body and "confusion_matrix" in body
