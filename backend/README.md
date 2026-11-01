---
title: OralSight API
emoji: 🦷
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Uncertainty-aware oral lesion screening API (research prototype)
---

# OralSight API

FastAPI service behind the OralSight mobile app. It runs a 5-model ONNX ensemble on CPU with:

- a data-driven image-quality gate,
- an out-of-distribution check,
- temperature-scaled scores with abstention,
- class activation map (CAM) heatmaps.

It's a research prototype from an AI in Healthcare lab. It is not a medical device.

The YAML header above is the Hugging Face Spaces configuration, so this folder can be pushed to a Docker Space as-is. See the repository root README for full setup and deployment steps.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Status and model version |
| `POST /analyze` | Analyse an uploaded image (multipart `file`; `?explain=false` skips the heatmap) |
| `GET /samples` / `POST /samples/{id}/analyze` | Held-out demo cases |
| `GET /metrics` | Saved test-set evaluation for the app's evidence screen |
| `GET /model-card` | Deployed model configuration |
| `GET /docs` | Interactive OpenAPI docs |

Training data: Chandrashekar H S et al. (2021), Oral Images Dataset, Mendeley Data V2, doi:10.17632/mhjyrn35p4.2 (CC BY 4.0).
