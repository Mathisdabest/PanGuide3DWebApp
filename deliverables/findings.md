## Tumor localization and review support
**Recorded Demo Results • cached from a completed real model run**

**Examination overview** · CT · original dimensions (X,Y,Z): (512, 512, 41) · 41 slices · spacing (0.816, 0.816, 5.0) mm.

**Pancreas localization** · predicted mask volume 79.46 cm³.

**Candidate lesion segmentation** · 8.23 cm³ · 2 connected predicted regions.

**Geometric intersection** · 4.56 cm³ · 55.4% of tumor mask inside pancreas mask · 5.7% of pancreas mask intersected. Mathematical overlap is not evidence of malignancy.

### Tumor segmentation confidence
**Mean model score across the predicted tumor mask: 91.2%**

Uncalibrated model score; not the probability that the region is cancerous. High scores can still be wrong.

| Region (largest first) | Mean model score | Voxels |
|---|---|---|
| 1 | 91.8% | 2,296 |
| 2 | 82.9% | 174 |

Scores average the network's tumor-channel sigmoid outputs, linearly resampled onto the original CT grid, within each displayed six-connected region. They describe the model's label preference, not diagnostic accuracy or a calibrated clinical risk.


### Scan-specific explanation
The model marked 2 candidate tumor region(s), totaling 8.23 cm³ of predicted mask volume. Their mathematical intersection with the predicted pancreas is 4.56 cm³ (55.4% of the candidate tumor mask). These views support inspection of location and spatial relationships during research review. The highlighted tissue is not confirmed cancer; this output cannot determine tumor type, stage, invasion, or treatment.


Model: PanGuide3D, 52,328,834 parameters · checkpoint 917acfa35774 · web software 0.3.0. Runtime: 109.837 seconds. Peak CUDA allocated memory: not measured.
