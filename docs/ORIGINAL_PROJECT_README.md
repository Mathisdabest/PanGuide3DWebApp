# PanGuide3D — pancreas CT segmentation and diagnostic support

Everything needed to serve **PanGuide3D** behind a web app: a scan goes in, and
pancreas head/body/tail regions, a tumor segmentation overlaid on them, and a
structured report with the reasons behind the assessment come out.

This folder is self-contained — it does not import from the research repo — so
it can be lifted into its own git repository.

> **PanGuide3D: Cohort-Robust Pancreas Tumor Segmentation via Probabilistic
> Pancreas Conditioning and a Transformer Bottleneck**
> Sunny Joy Ma, Xiang Ma. arXiv:[2604.20981](https://arxiv.org/abs/2604.20981)

---

## ⚠️ Read this first: the model is CT-only

PanGuide3D was trained and evaluated exclusively on **contrast-enhanced
abdominal CT**. Its preprocessing is a Hounsfield-unit window, a quantity that
only CT produces. **MRI has no Hounsfield units**, so an MR volume pushed
through this pipeline is normalized meaninglessly and the model will return a
confident, wrong answer rather than an error.

There is no MRI anywhere in this project — both cohorts (PanTS and MSD
Task07) are CT, and the raw files on disk are literally named `ct.nii.gz`.
A web app built on this model must **accept CT and reject MR**.
`preprocess.looks_like_ct()` is provided for exactly that gate.

> **Not a medical device.** Research software for decision support. Not
> validated for clinical use, not cleared by any regulator, and not a
> substitute for a radiologist's read.

**Deploying this or handing it on?** Read
[`DEPLOYMENT.md`](DEPLOYMENT.md) first — it covers the dataset licensing (the
bundled demo scans are CC BY-NC-**ND**), how to transfer the repo and host the
209 MB checkpoint, measured inference cost (221 s and 3.5 GB on 2 CPU cores,
which rules serverless out), and what has to be on the page before the public
sees it.

---

## The model

An nnU-Net-style 3D U-Net with a shared encoder and **two** decoders:

```
CT volume ─► encoder ─► Transformer bottleneck ─┬─► pancreas decoder ─► probabilistic
                                                │                       pancreas map
                                                │                            │
                                                │                   (conditions the tumor
                                                │                    decoder at multiple
                                                │                         scales)
                                                └─► tumor decoder ◄──────────┘
                                                        └────────────► tumor logits
```

The pancreas decoder produces a probabilistic pancreas map; the tumor decoder
is conditioned on that map at multiple scales, steering it toward the organ
rather than searching the whole abdomen. A lightweight Transformer bottleneck
supplies long-range context.

The paper isolates both ideas by ablation: **2Decoder** drops the Transformer
bottleneck, **TransBNeck** keeps the bottleneck but uses a single decoder with
no pancreas conditioning. Each ablation loses accuracy, and the full model is
best on both cohorts.

---

## Published performance

From the paper, Table 2 — five-fold cross-validation, mean ± SD across folds.
Trained on the PanTS training split (7,200 train / 1,800 validation of 9,000
cases); evaluated on the PanTS official test set (901 cases, in-cohort) and on
MSD Task07 Pancreas (57 cases, out-of-cohort).

**PanTS test (in-cohort)**

| Model | Tumor Dice ↑ | Pancreas Dice ↑ | FP vol (cm³) ↓ | Tumor sens. ↑ | Patient sens. ↑ |
|---|---|---|---|---|---|
| nnU-Net | 0.202 ± 0.054 | 0.305 ± 0.090 | 16.970 ± 6.107 | 0.383 ± 0.043 | 0.394 ± 0.049 |
| RADFM | 0.255 ± 0.042 | 0.559 ± 0.010 | 10.835 ± 0.338 | 0.491 ± 0.079 | 0.559 ± 0.011 |
| SwinUNETR | 0.336 ± 0.053 | **0.827 ± 0.009** | 9.883 ± 0.452 | 0.565 ± 0.082 | 0.699 ± 0.057 |
| 2Decoder | 0.367 ± 0.029 | 0.708 ± 0.016 | 7.828 ± 0.820 | 0.663 ± 0.070 | 0.714 ± 0.089 |
| TransBNeck | 0.412 ± 0.027 | 0.683 ± 0.014 | 6.773 ± 0.699 | 0.692 ± 0.077 | 0.731 ± 0.048 |
| **PanGuide3D** | **0.460 ± 0.041** | 0.804 ± 0.013 | **3.070 ± 1.554** | **0.787 ± 0.073** | **0.819 ± 0.039** |

**MSD Task07 (out-of-cohort)**

| Model | Tumor Dice ↑ | Pancreas Dice ↑ | FP vol (cm³) ↓ | Tumor sens. ↑ | Patient sens. ↑ |
|---|---|---|---|---|---|
| nnU-Net | 0.066 ± 0.057 | 0.145 ± 0.076 | 19.646 ± 5.965 | 0.075 ± 0.065 | 0.068 ± 0.057 |
| RADFM | 0.154 ± 0.048 | 0.489 ± 0.033 | 8.666 ± 1.387 | 0.333 ± 0.123 | 0.333 ± 0.123 |
| SwinUNETR | 0.274 ± 0.061 | 0.761 ± 0.024 | 8.550 ± 1.542 | 0.619 ± 0.105 | 0.602 ± 0.102 |
| 2Decoder | 0.410 ± 0.075 | 0.772 ± 0.022 | 4.781 ± 1.595 | 0.737 ± 0.106 | 0.737 ± 0.106 |
| TransBNeck | 0.421 ± 0.077 | 0.749 ± 0.026 | 4.291 ± 1.515 | 0.772 ± 0.106 | 0.772 ± 0.106 |
| **PanGuide3D** | **0.475 ± 0.069** | 0.753 ± 0.033 | **2.004 ± 1.765** | **0.842 ± 0.088** | **0.842 ± 0.088** |

Machine-readable in [`reference/benchmarks.json`](reference/benchmarks.json)
(read at runtime by `report.py`) and
[`reference/paper_table2_results.csv`](reference/paper_table2_results.csv).

**What these numbers mean for a UI.** Tumor Dice around 0.46–0.48 means the
model *locates* a lesion but does not delineate it precisely — never present
predicted volumes or diameters as measurements. Sensitivity around 0.79–0.84
means roughly one tumor in five or six is missed, so a negative result does not
rule out cancer. The paper also analyses accuracy by tumor size and pancreatic
subregion, but publishes those only as a heatmap with no tabulated values, so
this package quotes no subregion-specific accuracy.

### Shipped weights

The checkpoint in `checkpoints/` is the best-scoring training run of the
PanGuide3D architecture found in the research repo's `seg_models/`. Selected by
validation tumor Dice over every comparable full-length run; its exact source
directory, epoch, SHA-256 and full training configuration are recorded in
[`checkpoints/panguide3d.json`](checkpoints/panguide3d.json).

| | |
|---|---|
| Architecture | PanGuide3D (shared encoder, 2 decoders, Transformer bottleneck) |
| Parameters | 52.3 M |
| Size | 209 MB (float32) |

It is a single training run, not the five-fold ensemble the paper evaluates, so
treat the table above as the published performance of the architecture rather
than a measurement of this particular file.

---

## Install

```bash
pip install -r requirements.txt
```

`monai` is optional but recommended: without it, inference falls back to a
uniform-average sliding window rather than the gaussian-weighted one.

### Get the weights

Too large for plain git (209 MB vs GitHub's 100 MB limit). See
[`checkpoints/README.md`](checkpoints/README.md) for Git LFS setup or how to
re-export them.

---

## Use

```python
from panguide3d import PanGuide3D

analyzer = PanGuide3D()                    # load once, at app startup
result = analyzer.analyze("upload.nii.gz") # or a DICOM series directory

report = result.report
report.tumor_detected        # True / False
report.assessment            # "Tumor suspected in the pancreatic head"
report.confidence            # high / moderate / low
report.region_volumes_cm3    # {"head": 31.2, "body": 24.8, "tail": 19.4}
report.findings              # per-lesion volume, diameter, probability, region
report.reasons               # what drove the assessment
report.cautions              # what would make it unreliable
report.to_json()             # the whole thing, ready for an HTTP response
```

Masks for the viewer:

```python
result.prediction.pancreas_mask   # (Z, Y, X) uint8
result.prediction.tumor_mask      # (Z, Y, X) uint8
result.prediction.tumor_prob      # (Z, Y, X) float32 — for a heatmap
result.split.head / .body / .tail # (Z, Y, X) bool
result.split.label_map()          # 0=bg 1=head 2=body 3=tail, one-pass rendering
```

Rendering:

```python
from panguide3d.visualize import overlay_slice, best_tumor_slice, render_views

z = best_tumor_slice(result.prediction.tumor_mask, "axial")
rgb = overlay_slice(result.volume.array, z, "axial",
                    split=result.split, tumor_mask=result.prediction.tumor_mask)
# float RGB in [0,1] -> (rgb * 255).astype("uint8") for PIL

render_views(result.volume.array, split=result.split,
             tumor_mask=result.prediction.tumor_mask,
             out_dir="out", case_id=report.case_id)   # three MIP PNGs
```

Command line:

```bash
python scripts/run_inference.py --input case.nii.gz --out_dir out/
```

### Accepted inputs

| Input | How |
|---|---|
| NIfTI (`.nii`, `.nii.gz`) | `analyzer.analyze("scan.nii.gz")` |
| DICOM series directory | `analyzer.analyze("/path/to/series/")` — one series per directory |
| Volume already in memory | `analyzer.analyze_array(hu_array_zyx, spacing_xyz)` — raw HU |
| Research `.npz` | already preprocessed; see `scripts/make_examples.py` |

Four of each are bundled in `samples/` to try immediately.

Raw inputs go through the full recipe (resample, window, scale); the `.npz`
files in the research repo have had it applied already.

---

## Examples

`samples/` holds four worked PanTS test cases, each with its report JSON and
three overlay renders:

| Case | Shows |
|---|---|
| `PanTS_00009473` | tumor in the pancreatic **head** |
| `PanTS_00009005` | tumor in the pancreatic **body** |
| `PanTS_00009001` | tumor in the pancreatic **tail** |
| `PanTS_00009160` | **no tumor** — a negative case |

The scans themselves are bundled under `samples/<case>/input/`, so the
showcase is self-contained — nothing here needs the research cluster:

```bash
python scripts/make_examples.py                       # regenerate all four

python scripts/run_inference.py \
  --input samples/PanTS_00009473/input/ct.nii.gz \
  --out_dir out/                                      # one scan, upload path
```

Every case is run twice, from the raw `ct.nii.gz` and from the preprocessed
`.npz`, so the two input paths can be compared — they agree on all four. See
[`samples/README.md`](samples/README.md), and check the PanTS licence before
making the repository public.

These were chosen to be legible demonstrations that the pipeline works, not a
random sample — they are not a performance estimate.

---

## How it works

1. **Preprocess** (`preprocess.py`) — resample to 1.5 mm isotropic (B-spline),
   clip to the [-125, 275] HU abdominal window, rescale to [0, 1]. Fixed: it is
   what the weights were trained on, and deviating degrades predictions
   silently.
2. **Segment** (`predict.py`) — gaussian sliding window over 160×160×96 patches
   at 0.5 overlap. Channel 0 is pancreas, channel 1 is tumor. Sigmoid,
   threshold 0.5, then drop tumor components under 100 voxels as noise.
3. **Regionalize** (`regions.py`) — PCA the pancreas mask's long axis, cut into
   equal thirds, orient so the head is at the patient's right.
4. **Report** (`report.py`) — per-lesion volume, diameter and probability;
   region assignment; and the reasons and cautions behind the assessment.

### Head/body/tail is geometric, not anatomical

The network segments the pancreas as one object — it predicts no head/body/tail
labels. The subdivision splits the segmented organ into equal thirds along its
principal axis, the same construction the research project used to stratify
results by subregion.

Real anatomical boundaries are landmark-based (the neck lies over the superior
mesenteric vein), and equal thirds will not match them case by case. **The UI
should say so.** `regions.REGION_CAVEAT` holds wording you can display, and
`report.cautions` already includes it.

Which end is the head is decided anatomically: the pancreatic head lies to the
patient's right, so the axis is oriented to run right → left. `preprocess`
reorients every scan it reads to RAS first, which is what makes "the patient's
right" well defined in the array no matter how the upload was stored. That is
not hypothetical — the PanTS scans on disk are a mix of R-first and L-first
orientations.

**How accurate is the split?** PanTS ships its own `pancreas_head` /
`pancreas_body` / `pancreas_tail` segmentations, so this can be measured rather
than assumed. Over 58 lesion cases the equal-thirds split names the correct
subregion **48 times (82.8%)**. Roughly one lesion in six is therefore
attributed to a neighbouring region, which the UI should say plainly. (Taking
the raw PCA sign without reorienting, as the research code does, scores 79.3%.)
Reproduce with `scripts/validate_region_split.py`.

One consequence worth knowing: because this checkpoint was trained on that
mixed-orientation data *without* canonicalization, it is not mirror-invariant,
and reorienting an L-first scan can change its prediction relative to the
research pipeline. Reorienting is still correct — head and tail are
meaningless without knowing left from right — but predictions on L-first scans
will not reproduce the research repo's exactly.

A tumor near a boundary is handled two ways: `finding.region` names the region
holding its centroid, and `finding.region_overlap` gives its share in each
region, which is the more honest thing to show.

---

## Limitations

- **CT only.** No MRI, ever. See the warning at the top.
- **Dice ≈ 0.46–0.48**: the model locates lesions, it does not measure them.
- **Sensitivity ≈ 0.79–0.84**: roughly one tumor in five or six is missed. A
  negative does not rule out cancer.
- **False positives are routine** — about 3 cm³ per case in-cohort, 2 cm³
  out-of-cohort for PanGuide3D, and far more for every baseline. Every finding
  needs a human read.
- **Lesions under 100 voxels (~0.34 cm³) are discarded.** The report flags it
  when this happens (`cautions`, `prediction.tumor_suppressed`).
- **Head/body/tail boundaries are approximate by construction.**
- **No subregion-specific accuracy is published**, so none is claimed.
- **`report.confidence` is a heuristic** over lesion size and mean probability,
  not a calibrated risk score.
- **Shipped weights are one run, not the paper's five-fold ensemble.**

---

## Layout

```
panguide3d_webapp/
├── panguide3d/
│   ├── pipeline.py       PanGuide3D — the entry point
│   ├── preprocess.py     DICOM/NIfTI -> the tensor the model expects
│   ├── model.py          architecture config + checkpoint loading
│   ├── predict.py        sliding-window inference -> masks
│   ├── regions.py        head/body/tail split, tumor localization
│   ├── report.py         findings, reasons, cautions
│   ├── visualize.py      slice overlays and MIP renders
│   └── models/           the network definition (vendored verbatim)
├── checkpoints/          weights (not in git — see its README)
├── reference/
│   ├── benchmarks.json          paper Table 2, read by report.py
│   └── paper_table2_results.csv paper Table 2 as a table
├── LICENSE               MIT (code only — weights and scans carry dataset terms)
├── DEPLOYMENT.md         licensing, handover, hosting, public-demo checklist
├── samples/              worked demo cases, scans included
│   └── <case>/input/     the bundled ct.nii.gz and .npz
└── scripts/
    ├── run_inference.py        end-to-end CLI
    ├── make_examples.py        regenerate samples/
    ├── validate_region_split.py  measure the split against PanTS anatomy
    └── export_weights.py       re-export weights from a training checkpoint
```

## Provenance

Research repo: `/lustre/hdd/LAS/qli-lab/xiangm/knopan3d`. Model definitions are
vendored verbatim from `knopan3d_lib/models/` so the weights load exactly; the
inference path mirrors `analysis/infer.py`, and the region split matches
`analysis/stratify.py`. Reported performance is from arXiv:2604.20981, Table 2.

## Citation

```bibtex
@article{ma2026panguide3d,
  title   = {PanGuide3D: Cohort-Robust Pancreas Tumor Segmentation via
             Probabilistic Pancreas Conditioning and a Transformer Bottleneck},
  author  = {Ma, Sunny Joy and Ma, Xiang},
  journal = {arXiv preprint arXiv:2604.20981},
  year    = {2026}
}
```
