# OralSight

A mobile app for screening and reviewing oral lesion images. You take or upload a photo on your phone, and a hosted Python backend returns class scores, an image-quality check, an uncertainty (abstention) flag and a Grad-CAM heatmap. Built for an academic mini project: it does not stage cancer or give treatment advice.

## Repository layout

```text
mobile/     React Native (Expo) app + NativeWind (Tailwind): Person 2
backend/    FastAPI inference API: Person 2 (delivery) / Ralph (model)
ml/         Dataset, training, evaluation and model packaging: Ralph
  data/       local datasets (git-ignored)
  models/     model checkpoints and bundles (git-ignored)
  notebooks/  exploration / Colab notebooks
  scripts/    reproducible training and evaluation scripts
```

## Prerequisites

- Python 3.11+
- Node.js 22.13+ (Expo SDK 58 requirement)
- Android phone with **Expo Go** for development, or the built APK for the demo

## Running locally

**Backend.** Listen on all interfaces so a phone on the same Wi-Fi can reach it:

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pytest                      # API regression tests
```

Interactive API docs: <http://localhost:8000/docs>

**Mobile app:**

```bash
cd mobile
npm install
npx expo start              # scan the QR code with Expo Go
npm run typecheck && npm run lint
```

During development the app calls `http://<your-laptop-ip>:8000` automatically, using the Expo dev server's host. To point it somewhere else, set `EXPO_PUBLIC_API_URL` (see `mobile/.env.example`). On Windows, allow Python through the firewall on private networks, or the phone can't reach port 8000.

**ML environment** (Ralph):

```bash
cd ml
python -m venv .venv
pip install -r requirements.txt
```

## App screens

| Screen | What it does |
| --- | --- |
| Home | Take a photo or choose one from the gallery (with crop), photo tips, server status |
| Analysis | Preview, then analyse or pick another image. Shows the prediction, class scores, quality warnings, the abstention notice, the original/heatmap toggle and PDF export via the share sheet |
| Model evidence | Saved evaluation metrics, baseline vs fine-tuned comparison, confusion matrix and limitations, served from `GET /metrics` |

## API contract

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status, whether the model is loaded, model version |
| `POST /analyze` | Multipart `file` (JPEG/PNG/WebP, 10 MB max, at least 64 px per side) |
| `GET /metrics` | Contents of `backend/app/data/metrics.json` (written from Ralph's evaluation) |

`/analyze` response:

| Field | Meaning |
| --- | --- |
| `predicted_label` | Class label, or `null` if the model abstains |
| `class_scores` | Score for each class (after calibration) |
| `quality` | `status` (`ok`/`warning`/`reject`) and `reasons` |
| `abstained` | `true` when confidence is below the abstention threshold |
| `heatmap_available` / `heatmap_png_base64` | Grad-CAM overlay as a base64 PNG, if available |
| `model_version` | Version of the model bundle |
| `latency_ms` | Server-side processing time |
| `is_mock` | `true` until the real model is integrated |

Errors return `{"detail": "<message for the user>"}` with 413 (too large), 415 (not a readable or supported image) or 422 (empty or too small).

Uploads are validated by decoding them, not by trusting the client's content type. EXIF orientation is applied, and images are processed in memory and never stored.

**Integration point for Ralph:** [backend/app/model.py](backend/app/model.py). `Predictor.load()` runs once at startup, and `Predictor.predict()` takes an upright RGB PIL image. It's a mock for now. Ralph owns prediction semantics (labels, scores, abstention, quality rules); Person 2 owns delivering them through the API and the app. Keep [backend/app/schemas.py](backend/app/schemas.py) and [mobile/src/lib/types.ts](mobile/src/lib/types.ts) in sync.

## Deployment (free tier)

**Backend.** `backend/Dockerfile` runs on any free container host:

- **Hugging Face Spaces (recommended):** create a Docker Space and push `backend/`. The free CPU tier has 16 GB RAM, which is enough for PyTorch, and it serves over HTTPS. The container listens on port 7860.
- **Render (alternative):** a free web service built from the Dockerfile. It sleeps after 15 minutes idle and takes about a minute to wake, so open `/health` before the demo.
- **Vercel is not a good fit as-is.** Its Python functions have a 250 MB bundle limit, and PyTorch alone is bigger. It could only work if the model is exported to ONNX and served with `onnxruntime`.

**Mobile APK** (built in the cloud with EAS; a free Expo account is enough):

1. Set `EXPO_PUBLIC_API_URL` in `mobile/eas.json` (`preview` profile) to the deployed HTTPS backend URL.
2. Run `cd mobile && npx eas-cli@latest login`, then `npm run build:apk`.
3. Download the APK from the link EAS prints and install it on the phone (allow installs from unknown sources).

**Local fallback for the demo:** the APK allows plain HTTP, so you can build a second APK with `EXPO_PUBLIC_API_URL=http://<laptop-ip>:8000` and run the backend on the laptop over a phone hotspot.
