# Public demonstration dataset

MSD Task07 Pancreas, case `pancreas_045`, contributed by Memorial Sloan Kettering Cancer Center, New York, via the Medical Segmentation Decathlon.

Source: https://medicaldecathlon.com/ and https://medicaldecathlon.com/dataaws/
Dataset publication: Antonelli et al., “The Medical Segmentation Decathlon,” Nature Communications 13, 4128 (2022), https://doi.org/10.1038/s41467-022-30695-9
License: Creative Commons Attribution-ShareAlike 4.0, https://creativecommons.org/licenses/by-sa/4.0/
Download mirror: https://huggingface.co/datasets/MedOtter/msd-pancreas at revision 327dd55, imagesTr/pancreas_045.nii.gz.

The NIfTI is downloaded as provided by the mirror. Cached outputs, overlays, and CT-derived demo assets are adaptations for educational visualization and are supplied under CC BY-SA 4.0. Segmentation masks are predictions from the supplied PanGuide3D checkpoint, not reference annotations. Display uses a lossless LPS axis permutation/flip; inference uses the bundled preprocessing recipe. This is a selected demonstration case, not a performance estimate. The original project identifies this as a held-out fold-0 case; the actual training split file was not supplied, so held-out status has not been independently verified.

The MIT code license applies separately. No endorsement by the dataset contributors is implied. Local PanTS assets in samples/ are excluded from public packaging.
