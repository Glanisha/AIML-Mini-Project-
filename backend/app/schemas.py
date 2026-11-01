"""API contract shared between the backend and the mobile app.

Ralph owns prediction semantics (labels, scores, abstention);
Person 2 owns delivery of these fields over the API.
Keep in sync with mobile/src/lib/types.ts.
"""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class QualityReport(BaseModel):
    status: Literal["ok", "warning", "reject"]
    reasons: list[str]


class AnalyzeResponse(BaseModel):
    predicted_label: str | None  # None when the model abstains
    class_scores: dict[str, float]
    quality: QualityReport
    abstained: bool
    heatmap_available: bool
    heatmap_png_base64: str | None
    model_version: str
    latency_ms: float
    is_mock: bool
