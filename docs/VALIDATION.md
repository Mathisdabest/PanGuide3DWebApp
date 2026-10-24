# Verification performed on 2026-10-18

## Real checkpoint and CT

- Checkpoint SHA-256 matches the supplied metadata exactly; inspected ZIP and pickle globals statically, then loaded through PyTorch `weights_only=True` and strict key compatibility.
- Completed raw CT → validation → bundled preprocessing → actual model → masks → original-grid restoration on MSD Task07 pancreas_045.
- Device: Apple MPS GPU. Inference: **135.630 seconds**, measured with time.perf_counter and synchronized MPS completion. This excludes preprocessing, model loading, PDF and rendering. Original shape XYZ: **512×512×41**; model grid ZYX: **137×279×279**; restored output ZYX: **41×512×512**.
- Nonempty actual outputs: **23,842 pancreas voxels**, **2,470 tumor voxels** on original grid. Segmented volumes: approximately **79.46 cm³ pancreas**, **8.23 cm³ candidate tumor**. Intersection **4.56 cm³**. These are mask statistics, not clinical measurements or accuracy estimates.
- Process lifetime high-water RSS: **700,530,688 bytes** on macOS. This does not include an isolated GPU peak or represent all unified GPU memory. `gpu_peak_bytes` is null because CUDA measurement does not apply.
- See `demo-cache/MSD_pancreas_045.json` for machine-readable provenance and `.npz` for genuine arrays. The reference labels are not used to fabricate predictions.
- Two earlier CPU runs were interrupted after proving too slow; they are not marked passed and have no completed runtime measurements.

## Automated checks

` .venv/bin/python -m pytest -q `: **11 passed** on the final functional implementation (subsequent documentation-only updates do not affect that result; the final run output remains the authority).

Coverage: HU normalization, NIfTI read/spacing/origin/direction, physical landmark round-trip alignment, scalar 3D/CT validation, unsafe ZIP rejection, DICOM spatial sorting/intensity rescale/inconsistent-series rejection, omission of identifying output tags, physical volumes/true intersection, negative-result language, three-plane rendering and PDF format, rejection of a mismatched checkpoint before deserialization, consent, unavailable-GPU failure path, rate limit, and missing real cache refusal.

Synthetic arrays/DICOM are isolated test fixtures, not demonstration results. A full output-shape and nonempty-pancreas assertion is included in the real-cache smoke script; the full real run completed successfully.

## Browser and exports

Chromium 148 through Playwright: actual app loaded; four tabs present; dark theme verified; actual MSD NIfTI upload and CT preview completed; actual cached masks and findings loaded; both 3D surfaces visually verified. Fixed compatibility between Plotly 7 typed arrays and the Gradio frontend by serializing explicit coordinate/index lists. Re-rendering when the 3D tab opens prevents stale hidden-tab sizing.

Local HTTP root returned 200. PDF and three images generated from the actual masks. Final assets are under `deliverables/`. Short workflow capture and screenshots are under `recordings/`; the completed narrated MP4 is `deliverables/PanGuide3D-demo-3min.mp4`. Media verification: exactly 180 seconds, 1920×1080, H.264 at 30 fps with AAC narration; audio mean level −19.5 dB, peak −0.9 dB. Frames at 65, 110 and 140 seconds were extracted; CT overlays and rotating 3D surfaces were visually checked. Browser PDF download succeeded during recording.

## Public deployment

Deployed to https://huggingface.co/spaces/ilmuuu81/panguide3d. Hugging Face reported RUNNING on zero-a10g (the ZeroGPU API flavor), with no paid hardware selected. First anonymous Chromium test: HTTP 200, cached sample loaded, two mesh traces rendered, fresh real cloud inference succeeded, no browser errors. Model inference measured 3.168 seconds; browser end-to-end time was 10.526 seconds. Output differences from Apple MPS were small (79.44 vs 79.46 cm³ pancreas), as expected across numerical backends, and are not presented as exact bitwise agreement. Final provider-authenticated upload test passed: actual research NIfTI upload → inference → restored masks → report. End-to-end 7.334 seconds, model inference 3.079 seconds, peak CUDA allocated memory 2.824 GiB. See deliverables/public-verification.json. An additional anonymous upload test timed out, while the initial anonymous sample inference succeeded; the exact provider-side cause was not established. The authenticated upload succeeded using provider identity, not an application login requirement. GPU availability, per-identity quotas and queues still apply. A phone-sized Chromium layout at 390 pixels had 390-pixel content width (no horizontal overflow).

## Not verified

Exhausted quota/queue failure behavior on the provider (local failure paths are unit-tested), Safari/Edge and physical mobile devices, a 30-minute wall-clock file-expiration test, adversarial upload security audit, complete PHI detection, training-split provenance, independent model accuracy or clinical validation. Do not infer any of these from local tests. Owner logged in locally; source upload and ZeroGPU request for ilmuuu81/panguide3d completed. Hosted verification is tracked separately in public-verification.json when available.

Warnings observed: Python 3.9's LibreSSL/urllib3 support warning and library deprecation warnings (Gradio/FastAPI, MONAI/PyTorch indexing). No unit tests failed. Explicitly pinned torch 2.8 avoids the upcoming indexing semantic change. Use a supported Python 3.10 on Spaces and rerun validation there.

## Findings and reliability revision — 2026-10-22

13 automated tests passed, including real-mask intersection surfaces, empty-mask behavior, and evidence wording that cannot be mistaken for individual cancer probability. After the PDF symbol-format adjustment, all 9 core tests passed again. A real cached CT smoke test generated the three mesh layers, updated findings and PDF. Public Chromium verification confirmed cached real results, architecture metric display, the explicit unestablished diagnostic reliability label, the three named mesh layers and a successful PDF download, with no browser errors. See deliverables/review-update-verification.json. No new inference validation or diagnostic calibration is claimed for this presentation change. Paper Table 2 and calibration discussion were checked against https://arxiv.org/html/2604.20981v1; cancer education cites https://www.cancer.gov/types/pancreatic/diagnosis.

## Segmentation confidence — 2026-10-22

All 14 automated tests passed, including exact per-region means, voxel-weighted overall scoring, missing-probability and no-prediction behavior, and invalid input rejection. A new genuine MSD_pancreas_045 run completed on Apple MPS in 109.837 seconds. Original-grid candidate regions have 2,296 and 174 voxels; their mean sigmoid outputs are approximately 91.8% and 82.9%, with a voxel-weighted overall score of 91.2%. These are uncalibrated model scores, not cancer probabilities or accuracy measurements. Cached scores are from that same completed model run as the saved masks. The full probability map is used transiently and only region summary arrays are retained. Browser verification is saved in deliverables/confidence-verification.json when complete.
