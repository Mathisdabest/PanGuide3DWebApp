# Three-minute demonstration

Use 1920×1080 capture, browser at 100%, macOS Do Not Disturb enabled, and hide personal bookmarks. Use MSD_pancreas_045; do not use patient uploads. Open the app and verify Recorded Demo Results once before recording. Keep the actual result provenance visible. All candidate lesion references mean predicted masks, not cancer diagnosis.

| Time | Narration | Browser and mouse actions | Suggested annotation |
|---|---|---|---|
| 0:00–0:20 | “PanGuide 3D is an educational research tool for exploring experimental pancreas and candidate tumor segmentations in three-dimensional CT. It makes the model's spatial output easier to inspect. It is not a diagnostic system.” | Start on title and warning. Move slowly toward upload. | Research & Educational Use Only |
| 0:20–0:45 | “A CT scan is a volume with physical spacing and orientation. We use this openly licensed research example. The application validates the volume and applies the preprocessing specified by the supplied model.” | Accept notice. Choose MSD sample. Leave Recorded Demo Results checked, click Try Sample CT. | De-identified research CT · MSD Task07 |
| 0:45–1:10 | “These are cached outputs from a completed run of the actual trained checkpoint. Caching makes the demonstration independent of a cloud GPU queue. Live Inference runs the same model on a newly supplied scan.” | Point to cached provenance. Briefly indicate Live Inference button without claiming it ran. If recording a real live run separately, show actual wait or label time compression. | Previously computed real model results |
| 1:10–1:45 | “The green overlay marks predicted pancreas. Orange marks a candidate tumor region, when the model predicts one. The three planes show the same physical volume. We can inspect adjacent slices and change contrast and overlay opacity.” | Slowly move Z, Y and X sliders around starting focus. Adjust opacity .45→.2→.45; toggle tumor off/on. Avoid rapid movements. | Green: predicted pancreas · Orange: predicted tumor |
| 1:45–2:15 | “Surfaces reconstructed from these masks preserve physical proportions. Rotation helps us inspect the spatial relationship, but a visually convincing surface does not prove the segmentation is correct.” | Open 3D Visualization. Drag a smooth orbit, zoom slightly, toggle legend entries, restore preset. | Physical scale in millimeters |
| 2:15–2:40 | “The educational report is generated directly from the masks. It reports segmented volumes, connected regions and geometric intersection. These are model-output statistics, not validated clinical measurements or evidence of malignancy.” | Open AI Findings. Scroll slowly through volumes, intersection and limitations. Download PDF. | Deterministic, evidence-grounded report |
| 2:40–3:00 | “The contribution is an inspectable workflow from CT to experimental segmentation and visualization. False positives and missed lesions remain possible. Next steps include independent validation, clearer preprocessing provenance and testing the cloud deployment.” | Return to 3D or title; end with warning visible. | Future work: validation, robustness, cloud testing |

If the selected real case has no tumor prediction, say so and show that result; never add a lesion. Check the cache metrics before recording. A selected example is not evidence of general accuracy. Do not use architecture benchmark numbers as this checkpoint's evaluated accuracy.

## macOS recording

Press Shift–Command–5, select Record Selected Portion, choose the browser area, and select microphone in Options if narrating live. Record a short audio test. Start recording, follow the table, then stop from the menu bar. Trim with QuickTime Player (Edit → Trim). Save the original recording and an exported copy. The recording includes only the research example and no personal browser information.

## Optional automated capture

`scripts/record_demo.py` uses Playwright for a cached sample walkthrough, screenshots and a short browser video. Install Playwright separately (`.venv/bin/pip install playwright`, `.venv/bin/python -m playwright install chromium`), run the app, then `.venv/bin/python scripts/record_demo.py`. This is a capture aid, not a narrated three-minute film. It deliberately does not fabricate waits or trigger inference. The script's output only counts as recorded if it finishes successfully; see validation notes for whether it was run in this environment.

## Generated recording

The actual real-model cached walkthrough can be recorded with `scripts/record_full_demo.py` (180-second browser capture). Narration is in `scripts/narration.txt`; macOS `say -r 130 -f scripts/narration.txt -o recordings/narration.aiff` creates approximately 2:55 of synthesized narration. The final MP4, when present in deliverables/, combines that audio with the browser capture; it is a local cached-results demonstration, not a recording of cloud inference. The CT and masks come from the actual completed checkpoint run. The technical voice is synthesized; no audio from the original researchers is implied.
