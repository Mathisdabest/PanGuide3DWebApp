# Model card — PanGuide3D

**PanGuide3D: Cohort-Robust Pancreas Tumor Segmentation via Probabilistic
Pancreas Conditioning and a Transformer Bottleneck**
Sunny Joy Ma, Xiang Ma — arXiv:[2604.20981](https://arxiv.org/abs/2604.20981)
(q-bio.QM, submitted 22 April 2026)

## Overview

A 3D segmentation network that outputs two masks from one contrast-enhanced
abdominal CT volume: the pancreas, and pancreatic tumor. Head/body/tail regions
are derived geometrically afterwards; the network does not predict them.

- **Architecture** — nnU-Net-style 3D U-Net with a shared encoder, a pancreas
  decoder producing a probabilistic pancreas map, a tumor decoder conditioned
  on that map at multiple scales, and a lightweight Transformer bottleneck.
- **Parameters** — 52.3 M
- **Input** — one-channel CT volume, 1.5 mm isotropic (3.375 mm³ voxels)
- **Output** — two logit channels: 0 = pancreas, 1 = tumor
- **Patch size** — 160 × 160 × 96

The design targets *cohort robustness*: conditioning the tumor decoder on the
pancreas map constrains the search to the organ, which the paper reports
reduces anatomically implausible false positives and improves cross-cohort
generalization, particularly for small tumors.

## Training (as published)

| | |
|---|---|
| Training cohort | PanTS training split, 9,000 cases → 7,200 train / 1,800 validation |
| Held out | PanTS official test set, 901 cases |
| Evaluation | Five-fold cross-validation; each fold's checkpoint evaluated on both test sets |
| Loss | Dice + BCE |
| Sampling | Tumor-biased patch sampling and weighted sampling |
| Augmentation | Gaussian noise, blur, brightness/contrast, gamma, low-resolution simulation |
| Optimizer | AdamW by default (the nnU-Net baseline uses SGD with Nesterov momentum) |
| LR schedule | Polynomial decay, exponent 0.9 |
| Precision | Mixed precision; sliding-window validation inference |
| Hardware | 8 × NVIDIA A100 80 GB |

## Cohorts (paper, Table 1)

| Cohort | Cases | Tumor+ | Prevalence | Spacing | Mean tumor vol. |
|---|---|---|---|---|---|
| PanTS train | 7,200 | 684 | 0.095 | 1.5 mm | 2.59 cm³ |
| PanTS test | 901 | 151 | 0.168 | 1.5 mm | 2.64 cm³ |
| MSD Task07 | 57 | 57 | 1.000 | 1.5 mm | 7.12 cm³ |

MSD is entirely tumor-positive, so it measures delineation and sensitivity but
says nothing about specificity. PanTS test, at 17% prevalence, is the cohort
that exercises false positives.

## Results (paper, Table 2)

Mean ± SD across the five folds.

### PanTS test (in-cohort)

| Model | Tumor Dice ↑ | Pancreas Dice ↑ | FP vol (cm³) ↓ | Tumor sens. ↑ | Patient sens. ↑ |
|---|---|---|---|---|---|
| nnU-Net | 0.202 ± 0.054 | 0.305 ± 0.090 | 16.970 ± 6.107 | 0.383 ± 0.043 | 0.394 ± 0.049 |
| RADFM | 0.255 ± 0.042 | 0.559 ± 0.010 | 10.835 ± 0.338 | 0.491 ± 0.079 | 0.559 ± 0.011 |
| SwinUNETR | 0.336 ± 0.053 | **0.827 ± 0.009** | 9.883 ± 0.452 | 0.565 ± 0.082 | 0.699 ± 0.057 |
| 2Decoder | 0.367 ± 0.029 | 0.708 ± 0.016 | 7.828 ± 0.820 | 0.663 ± 0.070 | 0.714 ± 0.089 |
| TransBNeck | 0.412 ± 0.027 | 0.683 ± 0.014 | 6.773 ± 0.699 | 0.692 ± 0.077 | 0.731 ± 0.048 |
| **PanGuide3D** | **0.460 ± 0.041** | 0.804 ± 0.013 | **3.070 ± 1.554** | **0.787 ± 0.073** | **0.819 ± 0.039** |

### MSD Task07 (out-of-cohort)

| Model | Tumor Dice ↑ | Pancreas Dice ↑ | FP vol (cm³) ↓ | Tumor sens. ↑ | Patient sens. ↑ |
|---|---|---|---|---|---|
| nnU-Net | 0.066 ± 0.057 | 0.145 ± 0.076 | 19.646 ± 5.965 | 0.075 ± 0.065 | 0.068 ± 0.057 |
| RADFM | 0.154 ± 0.048 | 0.489 ± 0.033 | 8.666 ± 1.387 | 0.333 ± 0.123 | 0.333 ± 0.123 |
| SwinUNETR | 0.274 ± 0.061 | 0.761 ± 0.024 | 8.550 ± 1.542 | 0.619 ± 0.105 | 0.602 ± 0.102 |
| 2Decoder | 0.410 ± 0.075 | 0.772 ± 0.022 | 4.781 ± 1.595 | 0.737 ± 0.106 | 0.737 ± 0.106 |
| TransBNeck | 0.421 ± 0.077 | 0.749 ± 0.026 | 4.291 ± 1.515 | 0.772 ± 0.106 | 0.772 ± 0.106 |
| **PanGuide3D** | **0.475 ± 0.069** | 0.753 ± 0.033 | **2.004 ± 1.765** | **0.842 ± 0.088** | **0.842 ± 0.088** |

### Ablations

| Variant | What changes |
|---|---|
| **2Decoder** | Two decoders, no Transformer bottleneck |
| **TransBNeck** | Transformer bottleneck, single decoder, no pancreas conditioning |

Both ablations lose accuracy on both cohorts, and both roughly double
PanGuide3D's false-positive volume — the paper's evidence that the two
components contribute separately.

A two-stage U-Net was not reproduced as a baseline because its training code is
not released.

### Subgroup analysis

The paper analyses tumor Dice and sensitivity by tumor size and pancreatic
subregion (Figure 7) and Dice against tumor volume (Figure 6), but presents
both only graphically — no numeric values are tabulated. **This package
therefore quotes no subregion-specific or size-specific accuracy**, and
`report.py` deliberately does not claim any.

## Shipped checkpoint

The weights in `checkpoints/` are the best-scoring training run of this
architecture in the research repo's `seg_models/`, chosen by validation tumor
Dice across every comparable full-length run. Provenance — source directory,
epoch, SHA-256 and the complete training configuration — is recorded in
`checkpoints/panguide3d.json`.

It is **one training run, not the five-fold ensemble** the paper evaluates, so
the tables above describe the published architecture rather than measurements
of this specific file.

## Intended use

Decision support for a clinician reading an abdominal CT: pre-locate the
pancreas, subdivide it into regions, and flag candidate tumor for review.

**In scope** — contrast-enhanced abdominal CT of adults, acquired similarly to
PanTS or MSD Task07, with a trained reader reviewing every output.

**Out of scope**
- **MRI.** Never trained or evaluated on MR. The preprocessing is a
  Hounsfield-unit window, which MR does not produce. Reject MR uploads.
- Non-contrast CT, pediatric patients, post-surgical anatomy — untested.
- Autonomous diagnosis, screening asymptomatic populations, or any use without
  clinician review.
- Measuring lesion size for staging or response assessment. Dice ≈ 0.46–0.48
  means boundaries are approximate.

## Limitations

- **Misses roughly 1 tumor in 5–6** (tumor sensitivity 0.787 in-cohort, 0.842
  out-of-cohort). A negative result does not rule out cancer.
- **False positives are routine** — 3.07 cm³ per case in-cohort. Best among the
  models compared, but far from zero.
- **Dice ≈ 0.46–0.48**: lesions are located, not measured.
- **Head/body/tail boundaries are geometric** — equal thirds along the organ's
  principal axis, not landmark-based. Measured against PanTS's own anatomical
  subregion segmentations over 58 lesion cases, the split names the correct
  subregion 48 times (82.8%), so about one lesion in six is attributed to a
  neighbouring region. Reproduce with `scripts/validate_region_split.py`.
- **Lesions under 100 voxels (~0.34 cm³) are discarded** by this package's
  post-processing. Flagged in the report when it happens.
- **No published per-subregion accuracy**, so none is claimed.
- **`report.confidence` is a heuristic** over lesion size and mean probability,
  not a calibrated risk score. No calibration analysis is reported.
- **No fairness evaluation.** Neither cohort carries demographic metadata, so
  performance across age, sex and ethnicity is unknown.
- **Tumor prevalence in PanTS test is 17%**; a deployment population with
  different prevalence will see different positive predictive value even at
  identical sensitivity.

## Ethical considerations

Pancreatic cancer has a poor prognosis and delayed diagnosis is costly. Both
error directions carry real harm: a missed lesion delays treatment, and a false
positive can lead to unnecessary invasive workup. At these operating
characteristics the model must augment a radiologist's read, never replace or
pre-empt it.

Any deployment should log model version, input checksum and output alongside
the clinician's final read, so disagreements are auditable.

## Provenance

Reported performance: arXiv:2604.20981, Tables 1 and 2. Implementation vendored
from the research repo `knopan3d`
(`/lustre/hdd/LAS/qli-lab/xiangm/knopan3d`); inference mirrors
`analysis/infer.py` and the region split matches `analysis/stratify.py`.

Card prepared 2026-10-15.
