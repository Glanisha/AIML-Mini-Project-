// Mirrors backend/app/schemas.py — keep the two in sync.

export type QualityStatus = 'ok' | 'warning' | 'reject';

export interface QualityReport {
  status: QualityStatus;
  reasons: string[];
}

export interface AnalyzeResponse {
  predicted_label: string | null; // null when the model abstains
  class_scores: Record<string, number>;
  quality: QualityReport;
  abstained: boolean;
  heatmap_available: boolean;
  heatmap_png_base64: string | null;
  model_version: string;
  latency_ms: number;
  is_mock: boolean;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  malignant_recall: number;
  specificity: number;
  f1: number;
  roc_auc: number;
}

export interface MetricsResponse {
  is_mock: boolean;
  model_version: string;
  dataset: string;
  test_size: number;
  models: ModelMetrics[];
  confusion_matrix: { labels: string[]; matrix: number[][] };
  cpu_latency_ms: number | null;
  notes: string[];
}
