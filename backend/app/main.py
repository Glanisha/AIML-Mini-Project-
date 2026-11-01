import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .images import MAX_UPLOAD_BYTES, InvalidImageError, load_image
from .model import Predictor
from .schemas import AnalyzeResponse, HealthResponse

METRICS_PATH = Path(os.getenv("METRICS_PATH", Path(__file__).parent / "data" / "metrics.json"))

predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()  # load the model once, not per request
    yield


app = FastAPI(title="OralSight API", version="0.1.0", lifespan=lifespan)

# The native app does not need CORS; this only matters for browser tools like /docs
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=predictor.loaded, model_version=predictor.version)


@app.get("/metrics")
def metrics() -> dict:
    # Ralph's evaluation writes metrics.json; the app only displays saved results
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No evaluation metrics have been published yet.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    start = time.perf_counter()

    data = await file.read(MAX_UPLOAD_BYTES + 1)  # images are processed in memory and never stored
    try:
        image = load_image(data)
    except InvalidImageError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    prediction = await run_in_threadpool(predictor.predict, image)

    return AnalyzeResponse(
        predicted_label=prediction.predicted_label,
        class_scores=prediction.class_scores,
        quality=prediction.quality,
        abstained=prediction.abstained,
        heatmap_available=prediction.heatmap_png_base64 is not None,
        heatmap_png_base64=prediction.heatmap_png_base64,
        model_version=predictor.version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        is_mock=predictor.is_mock,
    )
