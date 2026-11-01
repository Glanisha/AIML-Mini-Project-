# OralSight: uncertainty-aware, explainable screening of oral lesion photographs

*AIML in Healthcare lab, mini project. Research prototype, not a medical device.*

## Abstract

Oral cancer is often diagnosed late, especially where specialists are scarce. We built OralSight, a mobile app and API that estimates whether a smartphone photo of an oral lesion resembles malignant or benign lesions. Every estimate comes with a calibrated score, an ensemble-based uncertainty estimate, a class-activation heatmap, and an explicit "no result" when the input or the model is unreliable.

We trained it on the public Oral Images Dataset: 323 original images, 158 malignant and 165 benign. A deep ensemble of five EfficientNet-B0 models, one per grouped cross-validation fold, was evaluated once on a locked test set of 65 images that had been held out before any training.

- **Discrimination:** ROC-AUC 0.939 (95% CI 0.875–0.991).
- **At the screening threshold:** sensitivity 93.8% (84.4–100) and specificity 78.8% (63.6–90.9). The threshold was chosen on cross-validation to target ≥90% sensitivity.
- **Abstention:** the system declines 18% of test cases. On the remaining 82% it reaches 96% sensitivity and 89% specificity.
- **Familiarity check:** it rejected 100% of non-oral photographs and synthetic images and none of the genuine test lesions.
- **Image-quality gate:** it turns severe blur, darkness, overexposure and sensor noise into abstentions instead of silent errors.
- **Runtime:** the full pipeline runs in about 80 ms on two CPU threads, and the server fits in a free hosting tier (335 MB RAM).

Fine-tuning was not significantly better than a frozen-feature baseline (ΔAUC +0.014, 95% CI −0.046 to +0.082). We argue that, at this data scale, the clinical value comes mainly from the safety layers around the classifier.

## 1. Problem and scope

**Task.** Binary classification of a photograph of an oral lesion as *benign-like* or *malignant-like*, presented as screening triage: "higher suspicion, seek prompt examination", "lower suspicion", or "uncertain".

**Out of scope, by design:**

- diagnosis
- cancer staging
- treatment advice
- healthy-mouth or precancerous classes (there are no labelled data for them)

The plan treated a larger dataset as an unresolved dependency. A 3,000-image Sri Lankan dataset with healthy and potentially-malignant classes exists (Piyarathne et al., 2024), but access requires approval, so it was not used.

## 2. Data

| Property | Value |
| --- | --- |
| Source | Chandrashekar et al. (2021), *Oral Images Dataset*, Mendeley Data V2, doi:10.17632/mhjyrn35p4.2. CC BY 4.0. Images collected with clinicians at hospitals in Karnataka, India, using mobile and intra-oral cameras |
| Used | `original_data` only: 165 benign, 158 malignant (all JPEG) |
| Excluded | The provided `augmented_data` folder (flips and rotations of the originals). Using it would put near-copies of a test image into training |
| Resolution | 93–9,248 px wide (median 518) |
| Patient IDs | **Not available** |
| Duplicates | 0 exact; 2 near-duplicate pairs (perceptual dHash including mirror images, Hamming ≤ 6), each grouped; 0 duplicate pairs with conflicting labels |
| Splits | Stratified, grouped hold-out of 65 test images (33 benign / 32 malignant). 5 stratified, grouped CV folds on the remaining 258. Verified: no duplicate group crosses a split or fold |

Because there are no patient IDs, near-duplicate grouping is only a *proxy* for patient-level splitting. Different photos of the same patient's lesion could still fall on both sides of a split. This is the most important threat to validity (§6).

## 3. Methods

### 3.1 Pipeline

```text
photo ─▶ validation (decode, EXIF orientation, size) ─▶ quality gate ─▶ familiarity (OOD) check
      ─▶ 5-model ensemble ─▶ temperature scaling ─▶ referral threshold + abstention ─▶ triage + CAM heatmap
```

Everything after validation is in `backend/app/inference/` and uses only NumPy, Pillow and ONNX Runtime. The evaluation scripts import this same package, so **every reported test number comes from the code that serves the app**. We confirmed this by comparing scores from the training and serving environments (identical), and by checking ONNX against PyTorch (max |Δlogit| 6.2×10⁻⁶).

### 3.2 Models

- **Baseline (plan task R3).** Frozen ImageNet EfficientNet-B0 global-average-pooled embeddings plus L2-regularised logistic regression. C was chosen by 5-fold CV on log-loss.
- **Fine-tuned CNNs (plan task R4).** EfficientNet-B0 and MobileNetV3-Large, both pretrained on ImageNet. The head is replaced by dropout(0.3) plus a linear layer on global-average-pooled features.
  - **Training:** AdamW (backbone lr 2×10⁻⁴, head 2×10⁻³, weight decay 10⁻⁴), one-cycle schedule, 30 epochs, mixed precision, batch 24.
  - **Training-only augmentation:** random resized crop (scale 0.55–1), flips, ±20° rotation, colour jitter, occasional blur.
  - **Model selection:** each fold keeps its lowest-validation-loss epoch.
- **Inputs.** The whole image is resized to 224×224 with no centre crop, so off-centre lesions are not cut away.
- **Architecture selection** used out-of-fold (OOF) AUC on development data only. EfficientNet-B0 scored 0.903 and MobileNetV3 0.899.
- **Deployment.** The five fold models form a deep ensemble (Lakshminarayanan et al., 2017). Their disagreement is a second uncertainty signal.

### 3.3 Calibration and decisions (fitted on OOF predictions, never on test)

- **Temperature scaling** (Guo et al., 2017): T = 1.25, fitted by minimising OOF negative log-likelihood.
- **Referral threshold:** the highest threshold whose OOF sensitivity is ≥ 90%, which gives 0.276 (OOF sensitivity 90.5%, specificity 72.0%). Screening is deliberately asymmetric: a missed malignancy costs more than an extra examination.
- **Abstention:** the system abstains when the score is within ±0.114 of the threshold (sized for about 85% OOF coverage) or when fewer than 60% of ensemble members agree.

### 3.4 Safety layers

- **Quality gate.** Four measurements are taken on a 256 px grayscale copy:
  - variance of the Laplacian (sharpness),
  - mean luminance,
  - share of clipped dark and bright pixels,
  - Immerkær's (1996) fast noise-σ estimate.

  Thresholds are percentiles of the *training* images. For example, an image is rejected if its sharpness is below half the 1st percentile. Noise estimation was added after the first evaluation showed that additive noise *inflates* the sharpness score and slipped past the blur check (§4.6).

- **Familiarity (OOD) check.** This uses deep k-nearest-neighbour detection (Sun et al., 2022): cosine distance from the L2-normalised frozen ImageNet embedding to its 5th-nearest development image. The threshold (0.522) is the 99th percentile of held-out-fold distances. *Disclosure:* the first version used Mahalanobis distance in a 32-dimensional PCA subspace. It caught only 0.5% of non-oral photos, because PCA throws away exactly the directions in which out-of-distribution images differ. We compared alternatives using the OOD proxy sets (which makes those sets no longer fully independent), and k-NN detected 100% while flagging 0% of genuine test lesions.
- **Explanation.** With a global-average-pool and linear head, Grad-CAM on the last convolutional block reduces exactly to class activation mapping (Zhou et al., 2016). We therefore compute the map *inside* the ONNX graph, with no autograd at serving time. The maps are averaged across ensemble members after per-member normalisation.

### 3.5 Evaluation

- **Locked test set, used once:** stratified bootstrap 95% CIs (2,000 resamples) and a paired bootstrap for the AUC difference against the baseline.
- **Calibration:** expected calibration error (ECE, 10 bins), Brier score and a reliability diagram.
- **Selective prediction:** risk–coverage curve and its area (AURC).
- **Clinical usefulness:** decision-curve analysis of net benefit (Vickers & Elkin, 2006).
- **Robustness:** synthetic blur, JPEG compression, darkening, brightening and sensor noise at three severities each.
- **OOD detection:**
  - 200 Oxford Flowers-102 photos (far-OOD natural images),
  - 60 synthetic patterns (noise, flat colour, gradients, checkerboards).
- **CPU latency:** ONNX Runtime.

## 4. Results (locked test set, n = 65)

### 4.1 Headline performance at the deployed threshold (0.276)

| Metric | Value | 95% CI |
| --- | --- | --- |
| ROC-AUC | **0.939** | 0.875–0.991 |
| PR-AUC | 0.933 | 0.864–0.990 |
| Sensitivity (malignant) | **93.8%** (30/32) | 84.4–100 |
| Specificity | 78.8% (26/33) | 63.6–90.9 |
| PPV | 81.1% | 71.4–91.2 |
| NPV | 92.9% | 83.3–100 |
| Balanced accuracy | 86.3% | 77.2–93.9 |

For reference, at the conventional 0.5 threshold, sensitivity would be only 75% and specificity 90.9%. That is why the sensitivity-targeted operating point matters for screening.

### 4.2 Model comparison

| Model | ROC-AUC | Sens. | Spec. |
| --- | --- | --- | --- |
| Frozen ImageNet embeddings + logistic regression (threshold 0.5) | 0.925 | 78.1% | 87.9% |
| MobileNetV3-Large ensemble (uncalibrated, 0.5) | 0.916 | 81.3% | 84.8% |
| EfficientNet-B0 ensemble (uncalibrated, 0.5) | 0.938 | 75.0% | 90.9% |
| **Deployed OralSight** (calibrated, threshold 0.276) | **0.939** | **93.8%** | 78.8% |

The paired-bootstrap AUC difference against the baseline is +0.014 (95% CI −0.046 to +0.082, one-sided p = 0.34): **not significant**. With 323 images, transfer-learned features already carry most of the signal.

### 4.3 Selective prediction

The model abstains on 12/65 test images (18%). Every abstention included "score too close to the threshold", and four also had ensemble disagreement. On the 53 answered cases:

- accuracy 92.5%,
- sensitivity 96.2% (25/26),
- specificity 88.9% (24/27).

AURC is 0.071. The risk–coverage curve falls monotonically as coverage is reduced, so abstention removes errors rather than random cases.

### 4.4 Calibration

ECE fell from 0.155 to 0.132 after temperature scaling; Brier stayed at 0.118. The reliability diagram shows residual **under-confidence between 0.3 and 0.7**: observed malignancy rates there are higher than predicted. A likely cause is that T was fitted on single-model OOF logits, while deployment averages five models, which already softens the probabilities. With 2–5 test cases per mid-range bin, these estimates are noisy.

### 4.5 Clinical usefulness

The decision curve lies above both "refer everyone" and "refer no one" for all threshold probabilities from 0.05 to 0.8. Net benefit depends on prevalence, and this dataset's roughly 50% prevalence is far above real screening populations.

### 4.6 Robustness (AUC / abstention rate)

| Corruption | Mild | Moderate | Severe |
| --- | --- | --- | --- |
| Gaussian blur (σ 1 / 3 / 6) | 0.93 / 22% | 0.89 / 58% | 0.85 / 98% |
| JPEG quality (50 / 20 / 5) | 0.93 / 17% | 0.92 / 22% | 0.81 / 58% |
| Darken (×0.7 / 0.45 / 0.2) | 0.93 / 23% | 0.90 / 48% | 0.82 / 100% |
| Brighten (×1.3 / 1.7 / 2.5) | 0.92 / 18% | 0.90 / 42% | 0.85 / 66% |
| Sensor noise (σ 8 / 20 / 40) | 0.93 / 20% | 0.88 / 63% | 0.58 / 94% |

The clean baseline is AUC 0.94 with 18% abstention. As degradation grows, the system increasingly declines to answer, which is the intended failure mode. Before the noise estimator was added, severe noise dropped AUC to 0.58 while the app abstained on only 35% of images. JPEG artefacts are still not detected by the quality gate itself; abstention at q=5 comes from model uncertainty.

### 4.7 Out-of-distribution detection

| Set | AUROC | Rejected |
| --- | --- | --- |
| Oxford Flowers-102 photos (n = 200) | 1.000 | 100% |
| Synthetic patterns (n = 60) | 1.000 | 100% |
| Genuine test lesions (n = 65) | n/a | 0% (false rejections) |

These are *far*-OOD proxies. Near-OOD inputs, such as healthy mouths, teeth or skin, were not available and may not be caught.

### 4.8 Explanations and failure analysis

See `ml/reports/figures/gradcam_gallery.png`. For true positives, attention concentrates on the ulcerated or erythematous region. Failures are instructive:

- **False negative:** a malignant case scored 0.08 with all five members agreeing. Attention sits on the tongue, not on the lesion at the commissure.
- **False positives:** two of the three are visually alarming benign lesions (one bleeding post-extraction site, one diffuse erythema).

Heatmaps show model attention, not verified lesion boundaries.

### 4.9 Runtime

- **Laptop (Ryzen 5 7235HS, 2 threads):** 80 ms median for the full pipeline (quality gate, OOD check, 5 models and heatmap).
- **Container limited to 0.5 vCPU and 512 MB (Render free-tier equivalent):** about 0.35 s per request at 335 MB peak memory.
- **Bundle size:** 97 MB (six ONNX files of about 16 MB each).

## 5. System

- **Mobile app:** React Native (Expo SDK 58), installable as an Android APK.
- **Workflow:** consent → photo capture or gallery with crop → result → PDF export through the share sheet.
- **Result screen:** a triage card, a score gauge (referral threshold, uncertain band, each ensemble member as a dot), quality and familiarity status, a photo/heatmap toggle, and technical details.
- **Other screens:**
  - held-out **sample cases**, including a known error and a degraded photo,
  - on-device-only history,
  - an **evidence** screen that charts the saved test results (ROC, calibration, risk–coverage, decision curve, robustness).
- **Backend:** FastAPI on ONNX Runtime with no PyTorch dependency, packaged in a Docker image for free hosting (Hugging Face Spaces or Render). Images are processed in memory and never stored. The app has no accounts and no analytics.

## 6. Limitations

1. **Small, single-region dataset** (323 images from Karnataka, India), with no external validation. Performance on other populations, skin tones, devices and lesion types is unknown.
2. **No patient IDs.** Near-duplicate grouping cannot exclude different photos of the same patient across splits, so results may be optimistic.
3. **Wide confidence intervals** with n = 65. For example, the specificity CI spans 64–91%.
4. **Prevalence about 50%.** In real screening, PPV would be far lower and false referrals would dominate.
5. **Two classes only.** Healthy mucosa and potentially malignant disorders are not modelled. The OOD check is validated only against far-OOD images.
6. **Under-confidence in mid-range scores.** Calibration should be refitted on ensemble outputs once more data allows a separate calibration split.
7. **Iterative safety fixes.** The OOD method and noise check were revised after the first test-set evaluation, as disclosed in §3.4. The classifier, thresholds and all other test metrics were unaffected, but the OOD and robustness numbers are no longer strictly first-look results.
8. **No subgroup analysis.** The data has no demographic metadata.
9. **Lesion types missing from training.** The malignant class is essentially squamous-cell carcinoma (red, white, ulcerated lesions), and there are no pigmented lesions. An outside image of a pigmented palatal lesion, likely an oral mucosal melanoma, illustrates the consequences:
   - The familiarity check correctly accepted it as a mouth photo.
   - The five models scored it 6–37% (ensemble 25.8%, just under the 27.6% threshold), with only 2/5 voting malignant.
   - The CAM heatmap focused on the palate margins, not the lesion.
   - Its nearest training images were open-mouth palate views with red or white lesions.

   Abstention (threshold band plus ensemble disagreement) turned what would have been a false "lower suspicion" into "uncertain: get it checked". The safety layer helped, but it can't substitute for training data covering such lesions.

## 7. Ethics

- **Positioning:** framed as screening *support*, with persistent disclaimers and first-launch consent.
- **Thresholding:** biased towards sensitivity.
- **Uncertainty:** the system refuses rather than guesses when it is unsure, and an uncertain result is presented as "get it checked", never as reassurance.
- **Persistent-lesion advice:** benign-like results always repeat the standard advice to examine any lesion that persists for more than two weeks.
- **Privacy:** no images are retained, and history stays on the device.
- **Data licence:** the dataset is CC BY 4.0 and is credited in the app, the API and here.

## 8. Next steps

1. Request access to the Piyarathne et al. (2024) dataset for external validation and for healthy and OPMD classes.
2. Collect or obtain patient-level identifiers for proper grouped splitting.
3. Use a dedicated calibration split, or fit temperature on ensemble outputs.
4. Add a JPEG blockiness metric and a near-OOD test set.
5. Run a prospective usability study with health workers.

## References

- Chandrashekar H S, Geetha Kiran A, Murali S, Dinesh M S, Nanditha B R. *Oral Images Dataset*. Mendeley Data, V2, 2021. doi:10.17632/mhjyrn35p4.2
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.
- Immerkær J. Fast noise variance estimation. Computer Vision and Image Understanding 64(2), 1996.
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
- Piyarathne NS et al. A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders. Oral Oncology 156, 2024.
- Selvaraju RR et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV 2017.
- Sun Y, Ming Y, Zhu X, Li Y. Out-of-distribution detection with deep nearest neighbors. ICML 2022.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Medical Decision Making 26(6), 2006.
- Zhou B, Khosla A, Lapedriza A, Oliva A, Torralba A. Learning deep features for discriminative localization. CVPR 2016.

## Appendix: Model card (`efficientnet_b0-ens5-v1.0.0`)

| | |
| --- | --- |
| **Type** | Image classifier (benign-like vs malignant-like oral lesion), with an OOD check, a quality gate and abstention |
| **Architecture** | Deep ensemble of 5 EfficientNet-B0 models (ImageNet-pretrained, fine-tuned; one per grouped CV fold), with a GAP + linear head |
| **Serving** | ONNX Runtime on CPU (`backend/app/model_bundle/`, 97 MB). Class activation maps are computed in-graph |
| **Developed by** | OralSight team (AI in Healthcare lab mini project), 2026 |
| **Licence of training data** | CC BY 4.0 (Chandrashekar et al., 2021, doi:10.17632/mhjyrn35p4.2) |
| **Status** | Research prototype. **Not a medical device. Not for clinical use.** |

### Intended use

- **For:** demonstrating explainable, uncertainty-aware screening *support* on smartphone photos of visible oral lesions.
- **Output:** a triage suggestion of "higher suspicion: seek prompt examination", "lower suspicion", or "uncertain / photo not usable".
- **Users:** researchers and students. In a future validated form, possibly community health workers, always with clinical follow-up.

### Out-of-scope uses

- Diagnosis, staging, prognosis or treatment decisions.
- Healthy mouths, teeth, or potentially malignant disorders, which were not in the training data.
- Pigmented lesions (e.g. oral mucosal melanoma). The training data contains essentially none; see the failure case in §6.
- Populations or imaging devices that differ substantially from the training data, unless re-validated.
- Any use that would delay a clinical examination.

### Details

| | |
| --- | --- |
| Training data | §2 |
| Decision logic | §3.3 and §3.4 |
| Performance | §4 |
| Limitations and ethics | §6 and §7 |
