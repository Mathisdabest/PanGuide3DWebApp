"""Turn an uploaded scan into the exact tensor the model was trained on.

The shipped checkpoint was trained on MSD Task07 volumes prepared by
``scripts/msd_pancreas_to_npz.py`` in the research repo. That recipe is:

    1. read the volume with SimpleITK
    2. resample to 1.5 x 1.5 x 1.5 mm (B-spline for the image)
    3. clip to the abdominal CT window [-125, 275] HU
    4. rescale that window linearly to [0, 1]

Anything that deviates from this silently degrades the prediction, so the
whole recipe lives in one place here and is re-used by inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

#: Abdominal CT window the model was trained with, in Hounsfield units.
HU_MIN, HU_MAX = -125.0, 275.0

#: Target voxel size in mm, (x, y, z).
TARGET_SPACING = (1.5, 1.5, 1.5)


@dataclass
class Volume:
    """A preprocessed scan plus everything needed to map results back."""

    array: np.ndarray                      # (Z, Y, X) float32 in [0, 1]
    spacing: tuple                         # (sx, sy, sz) mm, == TARGET_SPACING
    origin: tuple                          # world origin of the resampled image
    direction: tuple                       # direction cosines of the resampled image
    original_spacing: tuple
    original_size: tuple
    case_id: str = "case"
    meta: dict = field(default_factory=dict)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx) * float(sy) * float(sz)

    @property
    def shape(self):
        return self.array.shape


def resample(image: sitk.Image, out_spacing=TARGET_SPACING, is_label: bool = False) -> sitk.Image:
    """Resample to an isotropic grid, matching the training preprocessing."""
    in_spacing = image.GetSpacing()
    in_size = image.GetSize()
    out_size = [int(round(in_size[i] * (in_spacing[i] / out_spacing[i]))) for i in range(3)]

    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(float(s) for s in out_spacing))
    f.SetSize(out_size)
    f.SetOutputOrigin(image.GetOrigin())
    f.SetOutputDirection(image.GetDirection())
    f.SetTransform(sitk.Transform())
    f.SetDefaultPixelValue(0)
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return f.Execute(image)


def window_scale(array: np.ndarray) -> np.ndarray:
    """Clip to the abdominal window and rescale it to [0, 1]."""
    array = np.clip(array.astype(np.float32), HU_MIN, HU_MAX)
    return (array - HU_MIN) / float(HU_MAX - HU_MIN)


def reorient(image: sitk.Image, code: str = "RAS") -> sitk.Image:
    """Reorient to a canonical axis order, RAS by default.

    Uploads arrive in whatever orientation the scanner or converter produced.
    Normalizing here means the array axes always mean the same anatomical
    thing downstream - in RAS the first axis increases toward the patient's
    right, which is what ``regions`` relies on to tell the pancreatic head
    from the tail. Without this a scan stored the other way round would have
    its head and tail silently swapped.
    """
    try:
        return sitk.DICOMOrient(image, code)
    except Exception:
        # Older SimpleITK, or a direction matrix that is not axis-aligned.
        return image


def read_image(path) -> sitk.Image:
    """Read a NIfTI file, or a directory holding one DICOM series."""
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path))
        if not series:
            raise ValueError(f"No DICOM series found in {path}")
        if len(series) > 1:
            raise ValueError(
                f"{len(series)} DICOM series in {path}; pass a directory holding exactly one."
            )
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(path), series[0]))
        return reader.Execute()
    return sitk.ReadImage(str(path))


def preprocess(path, case_id: str | None = None) -> Volume:
    """Read a scan from disk and apply the full training-time recipe."""
    image = read_image(path)
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D volume, got {image.GetDimension()}D")

    original_spacing = tuple(float(v) for v in image.GetSpacing())
    original_size = tuple(int(v) for v in image.GetSize())

    image = reorient(image)
    resampled = resample(image, TARGET_SPACING, is_label=False)
    array = window_scale(sitk.GetArrayFromImage(resampled))

    if case_id is None:
        case_id = Path(path).name.replace(".nii.gz", "").replace(".nii", "") or "case"

    return Volume(
        array=array,
        spacing=TARGET_SPACING,
        origin=tuple(float(v) for v in resampled.GetOrigin()),
        direction=tuple(float(v) for v in resampled.GetDirection()),
        original_spacing=original_spacing,
        original_size=original_size,
        case_id=case_id,
        meta={"source": str(path)},
    )


def preprocess_array(array_zyx: np.ndarray, spacing_xyz, case_id: str = "case") -> Volume:
    """Same recipe for a volume already in memory, in raw HU, ordered (Z, Y, X)."""
    image = sitk.GetImageFromArray(np.asarray(array_zyx, dtype=np.float32))
    image.SetSpacing(tuple(float(s) for s in spacing_xyz))
    # No direction information to go on, so the array is taken to be RAS
    # already - see reorient().
    resampled = resample(image, TARGET_SPACING, is_label=False)
    return Volume(
        array=window_scale(sitk.GetArrayFromImage(resampled)),
        spacing=TARGET_SPACING,
        origin=tuple(float(v) for v in resampled.GetOrigin()),
        direction=tuple(float(v) for v in resampled.GetDirection()),
        original_spacing=tuple(float(s) for s in spacing_xyz),
        original_size=tuple(int(v) for v in image.GetSize()),
        case_id=case_id,
    )


def looks_like_ct(image: sitk.Image) -> bool:
    """Cheap sanity check that an upload is CT, not MR.

    The model is CT-only: its normalization is a Hounsfield-unit window, which
    has no meaning for MR intensities. Checks the DICOM modality tag when it
    survived the conversion, and otherwise falls back to the intensity range,
    since CT carries air at roughly -1000 HU and MR is non-negative.
    """
    try:
        modality = image.GetMetaData("0008|0060").strip().upper()
        if modality:
            return modality == "CT"
    except Exception:
        pass
    array = sitk.GetArrayFromImage(image)
    return bool(np.percentile(array, 1) < -300)
