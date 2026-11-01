import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalyzeResponse, HealthResponse, QualityReport

MODEL_VERSION = "mock-0.0.0"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="OralSight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=False, model_version=MODEL_VERSION)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    # MOCK: returns a fixed response until Ralph's model bundle is integrated (task P4)
    start = time.perf_counter()
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported format. Use JPEG, PNG or WebP.")
    await file.read()  # images are not retained

    return AnalyzeResponse(
        predicted_label="benign",
        class_scores={"benign": 0.72, "malignant": 0.28},
        quality=QualityReport(status="ok", reasons=[]),
        abstained=False,
        heatmap_available=False,
        model_version=MODEL_VERSION,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        is_mock=True,
    )
