# Model card: OralSight `efficientnet_b0-ens5-v1.0.0`

| | |
| --- | --- |
| **Type** | Image classifier (benign-like vs malignant-like oral lesion), with an OOD check, a quality gate and abstention |
| **Architecture** | Deep ensemble of 5 EfficientNet-B0 models (ImageNet-pretrained, fine-tuned; one per grouped CV fold), with a GAP + linear head |
| **Serving** | ONNX Runtime on CPU (`backend/app/model_bundle/`, 97 MB). Class activation maps are computed in-graph |
| **Developed by** | OralSight team (AI in Healthcare lab mini project), 2026 |
| **Licence of training data** | CC BY 4.0 (Chandrashekar et al., 2021, doi:10.17632/mhjyrn35p4.2) |
| **Status** | Research prototype. **Not a medical device. Not for clinical use.** |

## Intended use

- **For:** demonstrating explainable, uncertainty-aware screening *support* on smartphone photos of visible oral lesions.
- **Output:** a triage suggestion of "higher suspicion: seek prompt examination", "lower suspicion", or "uncertain / photo not usable".
- **Users:** researchers and students. In a future validated form, possibly community health workers, always with clinical follow-up.

## Out-of-scope uses

- Diagnosis, staging, prognosis or treatment decisions.
- Healthy mouths, teeth, or potentially malignant disorders, which were not in the training data.
- Pigmented lesions (e.g. oral mucosal melanoma). The training data contains essentially none; see the failure case in REPORT.md §6.
- Populations or imaging devices that differ substantially from the training data, unless re-validated.
- Any use that would delay a clinical examination.

## Decision logic

| Step | Rule |
| --- | --- |
| Quality gate | Reject or warn on blur, under- or over-exposure, clipping, or noise, using percentile thresholds from the training images |
| Familiarity | Cosine distance to the 5th-nearest training embedding > 0.522 → "photo not recognised" |
| Score | Mean of the five members' temperature-scaled (T = 1.25) P(malignant) |
| Referral | Score ≥ 0.276 (chosen on cross-validation for ≥ 90% sensitivity) |
| Abstain | Score within ±0.114 of 0.276, fewer than 60% of members agree, quality rejected, or unfamiliar image |

## Performance (locked test set, n = 65; 95% bootstrap CIs)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.939 (0.875–0.991) |
| Sensitivity | 93.8% (84.4–100) |
| Specificity | 78.8% (63.6–90.9) |
| NPV / PPV | 92.9% / 81.1% |
| Coverage, and accuracy on answered cases | 82% and 92.5% (sensitivity 96.2%, specificity 88.9%) |
| ECE / Brier | 0.132 / 0.118 |
| OOD rejection, non-oral photos / synthetic / genuine lesions | 100% / 100% / 0% |
| CPU latency (2 threads) | about 80 ms, including the heatmap |

Full results, robustness and figures are in [REPORT.md](REPORT.md) and `ml/reports/`.

## Training data

- 323 original images (165 benign, 158 malignant) from hospitals in Karnataka, India, taken with mobile and intra-oral cameras.
- The provided augmented copies were excluded.
- There are no patient IDs; 2 near-duplicate pairs were grouped.
- No demographic or clinical metadata is available.

## Ethical considerations and caveats

- **Optimistic estimates are possible:** there's no patient-level split, the source is a single region, there's no external validation, and the test set is small.
- **PPV is overstated:** the dataset is about 50% malignant, so real-world PPV would be much lower.
- **Fairness is unassessed:** no subgroup analysis was possible.
- **Calibration:** scores are somewhat under-confident in the 0.3–0.7 range.
- **Heatmaps** show model attention, not lesion boundaries.
- **Uncertain cases** are framed as "get it checked", never as reassurance. Benign-like results always carry the advice to examine any lesion that persists for more than two weeks.
- **Privacy:** images are not stored server-side.
