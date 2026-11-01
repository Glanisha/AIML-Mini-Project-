# OralSight

A prototype for screening and reviewing oral lesion images. Built for an academic mini project: it does not stage cancer or give treatment advice.

## Repository layout

```text
backend/    FastAPI service (Person 2): /health, /analyze
frontend/   React + TypeScript + Vite + Tailwind UI (Person 2)
ml/         Dataset, training, evaluation and model packaging (Ralph)
  data/       local datasets (git-ignored)
  models/     model checkpoints and bundles (git-ignored)
  notebooks/  exploration / Colab notebooks
  scripts/    reproducible training and evaluation scripts
```

## Prerequisites

- Python 3.11+
- Node.js 20+

## Running locally

**Backend** (<http://localhost:8000>, interactive docs at `/docs`):

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend** (<http://localhost:5173>):

```bash
cd frontend
npm install
npm run dev
```

The dev server forwards `/api/*` to the backend, so `/api/health` reaches `GET /health`.

**ML environment** (Ralph):

```bash
cd ml
python -m venv .venv
pip install -r requirements.txt
```

## API contract (draft)

`POST /analyze` takes a multipart `file` (JPEG/PNG/WebP) and returns:

| Field | Meaning |
| --- | --- |
| `predicted_label` | Class label, or `null` if the model abstains |
| `class_scores` | Score for each class (after calibration) |
| `quality` | `status` (`ok`/`warning`/`reject`) and `reasons` |
| `abstained` | `true` when confidence is below the abstention threshold |
| `heatmap_available` | Whether a Grad-CAM overlay can be returned |
| `model_version` | Version of the model bundle |
| `latency_ms` | Server-side inference time |
| `is_mock` | `true` until the real model is integrated |

The response is a **mock** for now. The class labels and score meaning get filled in once the dataset audit (R1) is done. Ralph owns what the predictions mean; Person 2 owns delivering them through the API and showing them in the UI. The schema lives in [backend/app/schemas.py](backend/app/schemas.py).
