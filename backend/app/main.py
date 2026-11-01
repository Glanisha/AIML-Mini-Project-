import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .images import MAX_UPLOAD_BYTES, InvalidImageError, load_image
from .model import load_model
from .schemas import AnalyzeResponse, HealthResponse, OODReport, QualityReport, SamplesResponse, TriageReport, UncertaintyReport

APP_DIR = Path(__file__).parent
METRICS_PATH = Path(os.getenv("METRICS_PATH", APP_DIR / "data" / "metrics.json"))
SAMPLES_DIR = APP_DIR / "samples"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()  # load once, not per request
    yield


app = FastAPI(title="OralSight API", version="1.0.0", lifespan=lifespan)

# The native app does not need CORS; this serves the web build and /docs
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
if SAMPLES_DIR.exists():
    app.mount("/samples/images", StaticFiles(directory=SAMPLES_DIR), name="sample-images")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = state["model"]
    return HealthResponse(status="ok", model_loaded=not model.is_mock, model_version=model.version)


@app.get("/metrics")
def metrics() -> dict:
    # Written by ml/oralsight_ml/evaluate.py; the app only displays saved results
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation metrics have been published yet.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/model-card")
def model_card() -> dict:
    model = state["model"]
    if model.is_mock:
        raise HTTPException(status_code=404, detail="No trained model is installed.")
    keys = ["version", "created", "architecture", "classes", "positive_class", "calibration", "decision", "ood", "training_data", "input"]
    return {k: model.config[k] for k in keys} | {"ensemble_size": len(model.members)}


def _sample_index() -> dict:
    index = SAMPLES_DIR / "samples.json"
    if not index.exists():
        raise HTTPException(status_code=404, detail="No demo samples are available.")
    return json.loads(index.read_text(encoding="utf-8"))


@app.get("/samples", response_model=SamplesResponse)
def samples(request: Request) -> SamplesResponse:
    data = _sample_index()
    base = str(request.base_url).rstrip("/")
    items = [s | {"image_url": f"{base}/samples/images/{s['id']}.jpg"} for s in data["samples"]]
    return SamplesResponse(attribution=data["attribution"], note=data["note"], samples=items)


def _respond(image: Image.Image, start: float, explain: bool) -> AnalyzeResponse:
    model = state["model"]
    r = model.predict(image, explain=explain)
    return AnalyzeResponse(
        predicted_label=r.predicted_label,
        class_scores=r.class_scores,
        positive_class=model.positive_class,
        positive_probability=r.positive_probability,
        operating_threshold=r.operating_threshold,
        abstain_margin=r.abstain_margin,
        quality=QualityReport(status=r.quality_status, reasons=r.quality_reasons, metrics=r.quality_metrics),
        abstained=r.abstained,
        abstain_reasons=r.abstain_reasons,
        uncertainty=UncertaintyReport(**r.uncertainty.__dict__, member_probabilities=r.member_probabilities),
        ood=OODReport(**r.ood.__dict__),
        triage=TriageReport(**r.triage.__dict__),
        heatmap_available=r.heatmap_jpeg_base64 is not None,
        heatmap_jpeg_base64=r.heatmap_jpeg_base64,
        model_version=model.version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        timings_ms=r.timings_ms,
        is_mock=model.is_mock,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), explain: bool = True) -> AnalyzeResponse:
    start = time.perf_counter()
    data = await file.read(MAX_UPLOAD_BYTES + 1)  # processed in memory, never stored
    try:
        image = load_image(data)
    except InvalidImageError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    return await run_in_threadpool(_respond, image, start, explain)


@app.post("/samples/{sample_id}/analyze", response_model=AnalyzeResponse)
async def analyze_sample(sample_id: str, explain: bool = True) -> AnalyzeResponse:
    start = time.perf_counter()
    ids = {s["id"] for s in _sample_index()["samples"]}
    if sample_id not in ids:
        raise HTTPException(status_code=404, detail="Unknown sample.")
    image = load_image((SAMPLES_DIR / f"{sample_id}.jpg").read_bytes())
    return await run_in_threadpool(_respond, image, start, explain)
