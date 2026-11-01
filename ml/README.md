# OralSight ML pipeline

One reproducible pipeline takes you from the raw dataset to the model bundle the backend serves:

```text
audit ─▶ 5-fold CV training (2 architectures) ─▶ calibration & thresholds ─▶ ONNX export ─▶ test evaluation
```

| Stage | Module | Output |
| --- | --- | --- |
| Dataset audit, duplicate grouping, splits | `oralsight_ml/audit.py` | `reports/dataset_audit.json`, `reports/splits.csv` |
| Fine-tuning (EfficientNet-B0, MobileNetV3-Large) | `oralsight_ml/train.py` | `artifacts/<arch>/fold*.pt`, out-of-fold logits |
| Baseline, OOD statistics, temperature scaling, thresholds, quality limits | `oralsight_ml/calibrate.py` | `artifacts/calibration.json` |
| Model bundle (ONNX + config), with parity check | `oralsight_ml/export.py` | `backend/app/model_bundle/` |
| Locked test evaluation, robustness, OOD, latency, figures, demo pack | `oralsight_ml/evaluate.py`, `figures.py` | `reports/metrics_full.json`, `reports/figures/`, `backend/app/data/metrics.json`, `backend/app/samples/` |

## Setup

```bash
cd ml
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the CPU wheel
pip install -r requirements.txt
```

Download the [Oral Images Dataset](https://data.mendeley.com/datasets/mhjyrn35p4/2) (CC BY 4.0). Copy only the `original_data` images into:

```text
ml/data/raw/benign/      <- original_data/benign_lesions
ml/data/raw/malignant/   <- original_data/malignant_lesions
```

Don't use the dataset's `augmented_data` folder. Its images are flips and rotations of the originals, so they would leak between the train and test splits.

## Run

```bash
python run_pipeline.py               # everything (≈10 min on an RTX 3050; CPU works but is slower)
python run_pipeline.py --skip-train  # recalibrate / re-export / re-evaluate saved checkpoints
```

## Methodological safeguards

- **No patient leakage proxy.** The dataset has no patient IDs, so photos that are near-duplicates are grouped together and always kept in the same split or fold. Near-duplicates are found with a perceptual dHash, including mirrored copies, at Hamming distance ≤ 6.
- **Locked test set.** About 20% of the groups (stratified) are held out before any training. The test set is used once, by `evaluate.py`, after everything else is frozen.
- **Selection on development data only.** All of these come from 5-fold out-of-fold predictions on the development set:
  - the choice of architecture,
  - the temperature,
  - the referral threshold (at least 90% sensitivity),
  - the abstention band (about 85% coverage).

  The OOD threshold comes from held-out-fold distances (99th percentile), and the quality limits from percentiles of the training images.

- **Train/serve parity.**
  - Evaluation preprocesses images with `backend/app/inference/preprocess.py`, the same code the API runs.
  - Headline metrics come from `OralSightModel`, the class the API serves, running the exported ONNX bundle.
  - The ONNX export is checked numerically against PyTorch.
- **Honest uncertainty.**
  - Metrics carry 95% stratified-bootstrap confidence intervals (2,000 resamples).
  - The baseline comparison uses a paired bootstrap.
  - Calibration (ECE, Brier), risk–coverage and decision-curve analysis are reported alongside the headline metrics.
