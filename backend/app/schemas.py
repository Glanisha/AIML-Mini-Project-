"""API contract shared between the backend and the mobile app.

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
    metrics: dict[str, float] = {}


class UncertaintyReport(BaseModel):
    confidence: float
    member_agreement: float
    member_std: float
    score_range: tuple[float, float]
    member_probabilities: list[float]


class OODReport(BaseModel):
    in_distribution: bool
    score: float
    threshold: float


class TriageReport(BaseModel):
    level: Literal["refer", "routine", "indeterminate"]
    title: str
    message: str


class AnalyzeResponse(BaseModel):
    predicted_label: str | None  # None when the model abstains
    class_scores: dict[str, float]
    positive_class: str
    positive_probability: float
    operating_threshold: float
    abstain_margin: float
    quality: QualityReport
    abstained: bool
    abstain_reasons: list[str]
    uncertainty: UncertaintyReport
    ood: OODReport
    triage: TriageReport
    heatmap_available: bool
    heatmap_jpeg_base64: str | None
    model_version: str
    latency_ms: float
    timings_ms: dict[str, float]
    is_mock: bool


class Sample(BaseModel):
    id: str
    category: str
    title: str
    true_label: str
    image_url: str


class SamplesResponse(BaseModel):
    attribution: str
    note: str
    samples: list[Sample]
