---
title: PanGuide 3D
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: '3.10'
app_file: app.py
pinned: false
---

# PanGuide 3D

AI-Powered 3D Pancreatic CT Analysis. Research & Educational Use Only — Not for Clinical Diagnosis.

Use **Try Sample CT / Demo Mode** for recorded outputs from a completed real checkpoint run, or turn off Recorded Demo Results to recompute on cloud hardware. Accept the research and privacy notice first. Upload only de-identified research CT.

Outputs are experimental pancreas and candidate tumor segmentations, not cancer diagnoses. No stage, prognosis, or treatment is inferred. False positives and false negatives are possible.

Set Space hardware to **ZeroGPU** only if eligible. This repository does not enable paid hardware. Anonymous usage has provider quotas and queue limits. No application sign-in is required.

Temporary uploaded inputs and generated exports expire in approximately 31 minutes while the app is running. Metadata checks cannot guarantee de-identification, particularly burned-in annotations. This app has not been assessed as HIPAA-compliant.

Code: MIT. Public CT example and adapted assets: MSD Task07, CC BY-SA 4.0; see public-samples/ATTRIBUTION.md. Weights: supplied PanGuide3D research checkpoint trained on MSD Task07; dataset attribution and terms accompany the model. See MODEL_AUDIT.md for checkpoint identity, source discrepancies and unverified GPU performance. The model's separate weight license was not supplied; confirm the authors' distribution terms before making the weight file publicly downloadable.
