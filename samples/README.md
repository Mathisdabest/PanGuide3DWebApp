# PanTS CT examples

Four original CT files supplied with the PanGuide3D source package are included unchanged for noncommercial research and educational demonstrations:

| Case | Reference example |
|---|---|
| PanTS_00009473 | Tumor in pancreas head |
| PanTS_00009005 | Tumor in pancreas body |
| PanTS_00009001 | Tumor in pancreas tail |
| PanTS_00009160 | Tumor-free case |

Upload `samples/<case>/input/ct.nii.gz` in the app. These hand-picked examples are not a performance evaluation; current checkpoint predictions may differ from the inherited project's reports. `manifest.json` records sizes and SHA-256 hashes verified against the supplied files.

## Attribution and terms

PanTS: The Pancreatic Tumor Segmentation Dataset, Wenxuan Li and collaborators (2025). Copyright The Johns Hopkins University.

Source: https://github.com/MrGiovanni/PanTS
Paper: https://arxiv.org/abs/2507.01291
License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International, https://creativecommons.org/licenses/by-nc-nd/4.0/
Full original license and warranty disclaimer: [LICENSE-PanTS](LICENSE-PanTS).

The scans are copied without modification from the supplied source package. They are not covered by this repository's MIT code license. No endorsement by the dataset creators is implied. Preprocessed arrays, cached predictions, and overlay renders are excluded because the dataset license restricts sharing adapted material.
