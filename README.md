# OralSight

**Uncertainty-aware, explainable screening of oral lesion photographs.** It's a mini project for an AI in Healthcare lab, built as a React Native app with a FastAPI + ONNX backend and a reproducible PyTorch training pipeline.

You photograph a lesion and OralSight runs these steps:

1. checks photo quality,
2. checks that the image actually looks like an oral lesion photo,
3. runs a calibrated 5-model ensemble,
4. shows a referral-oriented triage with a heatmap and the ensemble's (dis)agreement,
5. **says "no result" instead of guessing** when it is unsure.

> Research prototype. Not a medical device and not a diagnosis. No cancer staging or treatment advice.

## Results (locked 65-image test set)

| | |
| --- | --- |
| ROC-AUC | **0.939** (95% CI 0.875–0.991) |
| Sensitivity / specificity at the screening threshold | **93.8% / 78.8%** |
| On the 82% of cases it answers | 92.5% accuracy, 96% sensitivity |
| Non-oral photos rejected / genuine lesions wrongly rejected | 100% / 0% |
| CPU latency, full pipeline including heatmap | about 80 ms (2 threads) |

The full write-up is in [docs/REPORT.md](docs/REPORT.md), with the model card in [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and figures in [ml/reports/figures/](ml/reports/figures/).

## Repository layout

```text
mobile/                 React Native (Expo SDK 58) app + NativeWind
backend/                FastAPI service (Docker, free-tier deployable)
  app/inference/        framework-free inference: quality gate, OOD, ensemble, CAM
  app/model_bundle/     trained ONNX ensemble + config (committed, 97 MB)
  app/samples/          held-out demo cases (CC BY 4.0)
  tests/                API regression tests (run against the real model)
ml/                     reproducible training & evaluation pipeline (PyTorch)
  oralsight_ml/         audit → train → calibrate → export → evaluate
  reports/              metrics, dataset audit, splits, figures
docs/                   research report and model card
render.yaml             optional Render blueprint
```

## Architecture

```text
 Phone (Expo app)                       Backend (FastAPI + ONNX Runtime, CPU)
 ────────────────                       ─────────────────────────────────────
 capture / gallery + crop ──multipart──▶ validate (decode, EXIF, size limits)
 consent, history (device only)          quality gate  (blur, exposure, noise)
 result: triage, gauge, heatmap  ◀─JSON─ familiarity check (deep k-NN on ImageNet features)
 evidence charts, model card             5× EfficientNet-B0 → temperature scaling
 PDF export (share sheet)                referral threshold + abstention → triage + CAM
```

## Running locally

**Prerequisites:** Python 3.11+, Node.js 22.13+, and an Android phone with Expo Go (or a browser).

**Backend** (the trained bundle is already in the repo):

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pytest                      # 18 API tests against the real model
```

API docs: <http://localhost:8000/docs>

**Mobile app:**

```bash
cd mobile
npm install
npx expo start              # scan the QR code with Expo Go
npm run web                 # or open http://localhost:8081 in a browser
npm run typecheck && npm run lint
```

In development the app calls `http://<laptop-ip>:8000` automatically. Set `EXPO_PUBLIC_API_URL` to override this (see `mobile/.env.example`). On Windows, allow Python through the firewall on private networks.

**Retraining** (optional; about 10 minutes on an RTX 3050): see [ml/README.md](ml/README.md).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status, whether a model is loaded, model version |
| `POST /analyze` | Multipart `file` (JPEG/PNG/WebP, 10 MB max, at least 64 px per side); `?explain=false` skips the heatmap |
| `GET /samples`, `POST /samples/{id}/analyze` | Held-out demo cases |
| `GET /metrics` | Saved test evaluation (powers the Evidence screen) |
| `GET /model-card` | Deployed model configuration |

The `/analyze` response contains:

- `triage` (`refer` / `routine` / `indeterminate`, with title and message)
- `predicted_label` (`null` when the model abstains), `class_scores`, `positive_probability`
- `operating_threshold`, `abstain_margin`
- `abstained`, `abstain_reasons`
- `uncertainty` (agreement, spread, each member's probability)
- `quality` (status, reasons, metrics), `ood` (in-distribution flag, score, threshold)
- `heatmap_jpeg_base64`
- `model_version`, `latency_ms`, `timings_ms`, `is_mock`

The schema lives in [backend/app/schemas.py](backend/app/schemas.py), mirrored in [mobile/src/lib/types.ts](mobile/src/lib/types.ts).

Errors return `{"detail": "<message for the user>"}` with status 413, 415 or 422.

## Deployment (free)

**Backend: Hugging Face Spaces (recommended).**

- The free CPU tier has 16 GB RAM, HTTPS, and sleeps only after long inactivity.
- Create a **Docker** Space and push the contents of `backend/`. Its `README.md` already contains the Space configuration (`sdk: docker`, `app_port: 7860`).
- The ONNX files are 16 MB each, and Spaces require Git LFS for files over 10 MB: run `git lfs track "*.onnx" "*.npz"` in the Space repo before committing them.
- The image is about 670 MB with no PyTorch dependency.
- Verified locally under Render-like limits (0.5 vCPU, 512 MB RAM): 335 MB peak memory and about 0.35 s per analysis.

**Backend: Render (alternative).** Use `render.yaml`, or create a free Docker web service with root directory `backend`. It sleeps after 15 minutes idle and takes about 1 minute to wake, so open `/health` before a demo.

**Vercel** isn't suitable for the backend, because Python functions there are capped at 250 MB.

**Android APK** (EAS cloud build, free Expo account):

1. Put the deployed HTTPS URL in `mobile/eas.json` → `build.preview.env.EXPO_PUBLIC_API_URL`.
2. Run `cd mobile && npx eas-cli@latest login && npm run build:apk`.
3. Install the APK from the link EAS prints. On the phone, allow installs from unknown sources.

**Offline fallback for the demo:** build a second APK with `EXPO_PUBLIC_API_URL=http://<laptop-ip>:8000` and run the backend on the laptop over a phone hotspot. Cleartext HTTP is enabled for this case.

## Data credit

Chandrashekar H S, Geetha Kiran A, Murali S, Dinesh M S, Nanditha B R (2021). *Oral Images Dataset*. Mendeley Data, V2. doi:10.17632/mhjyrn35p4.2. Licensed CC BY 4.0. Only the original images are used; samples bundled with the API are unmodified held-out images, plus one deliberately blurred copy labelled as such.
