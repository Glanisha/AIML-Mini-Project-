"""API contract shared between the backend and the React UI.

Ralph owns prediction semantics (labels, scores, abstention);
Person 2 owns delivery of these fields over the API.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class QualityReport(BaseModel):
    status: str  # "ok" | "warning" | "reject"
    reasons: list[str]


class AnalyzeResponse(BaseModel):
    predicted_label: str | None  # None when the model abstains
    class_scores: dict[str, float]
    quality: QualityReport
    abstained: bool
    heatmap_available: bool
    model_version: str
    latency_ms: float
    is_mock: bool
