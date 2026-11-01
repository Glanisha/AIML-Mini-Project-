// Mirrors backend/app/schemas.py — keep the two in sync.

export type QualityStatus = 'ok' | 'warning' | 'reject';
export type TriageLevel = 'refer' | 'routine' | 'indeterminate';

export interface QualityReport {
  status: QualityStatus;
  reasons: string[];
  metrics: Record<string, number>;
}

export interface UncertaintyReport {
  confidence: number;
  member_agreement: number;
  member_std: number;
  score_range: [number, number];
  member_probabilities: number[];
}

export interface AnalyzeResponse {
  predicted_label: string | null; // null when the model abstains
  class_scores: Record<string, number>;
  positive_class: string;
  positive_probability: number;
  operating_threshold: number;
  abstain_margin: number;
  quality: QualityReport;
  abstained: boolean;
  abstain_reasons: string[];
  uncertainty: UncertaintyReport;
  ood: { in_distribution: boolean; score: number; threshold: number };
  triage: { level: TriageLevel; title: string; message: string };
  heatmap_available: boolean;
  heatmap_jpeg_base64: string | null;
  model_version: string;
  latency_ms: number;
  timings_ms: Record<string, number>;
  is_mock: boolean;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
}

export interface Sample {
  id: string;
  category: 'malignant' | 'benign' | 'uncertain' | 'misclassified' | 'degraded' | string;
  title: string;
  true_label: string;
  image_url: string;
}

export interface SamplesResponse {
  attribution: string;
  note: string;
  samples: Sample[];
}

export interface Metric {
  value: number;
  ci95: [number, number];
}

export interface ReliabilityBin {
  mean_predicted: number;
  observed_rate: number;
  count: number;
}

export interface RobustnessRow {
  level: string;
  auc: number;
  sensitivity: number;
  specificity: number;
  abstain_rate: number;
  quality_flag_rate: number;
}

export interface MetricsResponse {
  is_mock: boolean;
  model_version: string;
  architecture?: string;
  dataset: string;
  audit?: {
    n_images: number;
    class_counts: Record<string, number>;
    near_duplicate_pairs: number;
    n_groups: number;
    patient_ids_available: boolean;
  };
  test_size: number;
  test_class_counts?: Record<string, number>;
  operating_threshold?: number;
  headline?: Record<'roc_auc' | 'pr_auc' | 'sensitivity' | 'specificity' | 'ppv' | 'npv' | 'balanced_accuracy' | 'f1', Metric>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
  selective?: { coverage: number; abstained: number; sensitivity?: number; specificity?: number; accuracy?: number };
  calibration?: {
    ece_before: number;
    ece_after: number;
    brier_before: number;
    brier_after: number;
    reliability_before: ReliabilityBin[];
    reliability_after: ReliabilityBin[];
  };
  comparison?: { name: string; key: string; roc_auc: number; sensitivity: number; specificity: number; accuracy: number }[];
  deployed_vs_baseline_auc?: { difference: number; ci95: [number, number]; p_value_one_sided: number };
  curves?: { roc: [number, number][]; roc_baseline: [number, number][]; pr: [number, number][] };
  risk_coverage?: { aurc: number; points: { coverage: number; risk: number; sensitivity: number }[] };
  decision_curve?: { threshold: number; model: number; treat_all: number; treat_none: number }[];
  robustness?: Record<string, RobustnessRow[] | (Omit<RobustnessRow, 'level'> & { level?: string })>;
  ood?: { threshold: number; in_distribution_test_flagged_rate: number; sets: Record<string, { n: number; auroc: number; detected_rate: number }> };
  cpu_latency_ms: number | null;
  notes: string[];
}
