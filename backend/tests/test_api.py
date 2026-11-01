import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs lifespan, so the model is loaded
        yield c


def make_image(fmt="JPEG", size=(320, 320), exif_orientation=None, colour=(190, 110, 110)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, colour)
    # Add texture so the quality gate sees a sharp image
    px = img.load()
    for x in range(0, size[0], 4):
        for y in range(0, size[1], 4):
            px[x, y] = (120, 60, 70)
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def post(client, data: bytes, name="photo.jpg", content_type="image/jpeg", **params):
    return client.post("/analyze", files={"file": (name, data, content_type)}, params=params)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_analyze_accepts_supported_formats(client, fmt):
    res = post(client, make_image(fmt))
    assert res.status_code == 200
    body = res.json()
    for key in ["predicted_label", "class_scores", "quality", "abstained", "abstain_reasons", "uncertainty", "ood", "triage", "model_version", "latency_ms"]:
        assert key in body
    assert body["heatmap_available"] == (body["heatmap_jpeg_base64"] is not None)
    assert abs(sum(body["class_scores"].values()) - 1) < 1e-3
    assert body["triage"]["level"] in {"refer", "routine", "indeterminate"}
    # An abstaining model must never return a label
    assert (body["predicted_label"] is None) == body["abstained"]


def test_explain_can_be_disabled(client):
    body = post(client, make_image(), explain="false").json()
    assert body["heatmap_jpeg_base64"] is None


def test_analyze_ignores_client_content_type(client):
    # Mobile clients sometimes send application/octet-stream; validation is by decoding
    assert post(client, make_image(), content_type="application/octet-stream").status_code == 200


def test_analyze_handles_exif_rotation(client):
    assert post(client, make_image(size=(300, 200), exif_orientation=6)).status_code == 200


def test_rejects_non_image(client):
    assert post(client, b"definitely not an image", name="notes.txt", content_type="text/plain").status_code == 415


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


def test_repeated_requests_are_deterministic(client):
    data = make_image()
    results = [post(client, data, explain="false").json()["class_scores"] for _ in range(4)]
    assert all(r == results[0] for r in results)


def test_flat_image_is_not_answered(client):
    # A featureless grey frame is blurry and unlike any lesion photo: the pipeline must abstain
    buf = io.BytesIO()
    Image.new("RGB", (320, 320), (128, 128, 128)).save(buf, format="PNG")
    body = post(client, buf.getvalue(), name="grey.png", content_type="image/png").json()
    assert body["abstained"] is True


def test_metrics_served(client):
    res = client.get("/metrics")
    assert res.status_code in (200, 404)
    if res.status_code == 200:
        assert "confusion_matrix" in res.json()


def test_samples_round_trip(client):
    res = client.get("/samples")
    if res.status_code == 404:
        pytest.skip("no sample pack installed")
    samples = res.json()["samples"]
    assert samples
    first = samples[0]
    assert client.get(first["image_url"].split("testserver")[-1]).status_code == 200
    body = client.post(f"/samples/{first['id']}/analyze").json()
    assert "triage" in body


def test_unknown_sample_404(client):
    res = client.post("/samples/does-not-exist/analyze")
    assert res.status_code == 404
