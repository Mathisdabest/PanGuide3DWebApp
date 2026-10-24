"""Split a pancreas mask into head, body and tail, and locate tumors in it.

The network segments the pancreas as one object; it is not trained on
head/body/tail labels. The subdivision is geometric, following the one the
research repo used to stratify its results (``analysis/stratify.py``), except
for how the axis is oriented (see below):

    1. take the pancreas voxels in millimetre coordinates
    2. find the organ's long axis with PCA
    3. cut the extent along that axis into equal thirds

Because the thirds are equal lengths rather than anatomical landmarks, the
boundaries are an approximation - see ``REGION_CAVEAT``.

**Which end is the head.** PCA fixes the long axis only up to sign, so
something has to decide which end is which. The anchor used here is
anatomical: the pancreatic head lies to the patient's right and the tail runs
off to the left, so the axis is oriented to run right -> left. ``preprocess``
reorients every scan it reads to RAS, which is what makes "the patient's
right" a well-defined direction in the array regardless of how the upload was
stored. That matters in practice - the PanTS scans on disk are a mix of R-first
and L-first orientations, and the research code, which takes the raw PCA sign
and never reorients, swaps head and tail on some of the L-first ones.

**How good the split is.** Measured against PanTS's own ``pancreas_head`` /
``pancreas_body`` / ``pancreas_tail`` segmentations over 58 lesion cases, this
rule names the correct subregion for 48 (82.8%); taking the raw PCA sign
instead scores 46 (79.3%). The residual error is the equal-thirds geometry
itself, which does not follow the anatomical landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGION_NAMES = ("head", "body", "tail")

REGION_CAVEAT = (
    "Head/body/tail boundaries are derived geometrically by splitting the "
    "segmented pancreas into equal thirds along its principal axis, not from "
    "anatomical landmarks such as the superior mesenteric vein. Checked "
    "against PanTS's own anatomical subregion labels on 58 cases, this split "
    "names the right subregion about 83% of the time, so roughly one lesion "
    "in six is attributed to a neighbouring region. Treat it as an "
    "approximate guide to location, not a definitive anatomical assignment."
)

#: Minimum pancreas voxels before a PCA split is meaningful.
MIN_PANCREAS_VOXELS = 10


@dataclass
class RegionSplit:
    """Head/body/tail masks plus the axis they were derived from."""

    head: np.ndarray | None
    body: np.ndarray | None
    tail: np.ndarray | None
    axis: np.ndarray | None          # unit vector in mm space, (x, y, z)
    centroid: np.ndarray | None      # pancreas centroid in mm, (x, y, z)
    bounds: tuple | None             # the four cut positions along the axis

    @property
    def available(self) -> bool:
        return self.head is not None

    def as_dict(self) -> dict:
        return {"head": self.head, "body": self.body, "tail": self.tail}

    def label_map(self) -> np.ndarray | None:
        """One array with 0=background, 1=head, 2=body, 3=tail."""
        if not self.available:
            return None
        out = np.zeros(self.head.shape, dtype=np.uint8)
        out[self.head] = 1
        out[self.body] = 2
        out[self.tail] = 3
        return out


def _axis_and_projection(pancreas_mask, spacing_xyz):
    """PCA long axis of the pancreas, oriented head-first, plus projections.

    SVD fixes the axis only up to sign, so which end comes out first is an
    artifact of the arithmetic rather than anatomy, and something has to
    choose. The anchor is anatomical: the pancreatic head lies to the
    patient's right and the tail runs off to the left, so the axis is oriented
    to run right -> left.

    ``preprocess`` reorients every scan it reads to RAS, where the first array
    axis increases toward the patient's right; head -> tail is therefore the
    direction of decreasing x. Arrays handed in directly (the research .npz
    files, for instance) are assumed to already be in that orientation.

    Anchoring on patient anatomy rather than on the mask's own shape matters:
    a shape-based rule - say, "the head is the bulkiest third" - reads
    differently off a ground-truth mask than off a predicted one, so the same
    case can change region between the reference and the prediction.
    """
    zz, yy, xx = np.where(pancreas_mask.astype(bool))
    if xx.size < MIN_PANCREAS_VOXELS:
        return None

    sx, sy, sz = (float(s) for s in spacing_xyz)
    coords = np.stack([xx * sx, yy * sy, zz * sz], axis=1).astype(np.float64)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]

    # Point the axis toward the patient's left, i.e. head -> tail.
    if axis[0] > 0:
        axis = -axis

    return (zz, yy, xx), centered @ axis, axis, centroid


def split_pancreas(pancreas_mask: np.ndarray, spacing_xyz) -> RegionSplit:
    """Divide the pancreas mask into head, body and tail thirds."""
    result = _axis_and_projection(pancreas_mask, spacing_xyz)
    if result is None:
        return RegionSplit(None, None, None, None, None, None)

    (zz, yy, xx), projection, axis, centroid = result

    # _axis_and_projection has already oriented the axis head -> tail.
    cuts = np.linspace(projection.min(), projection.max(), 4)

    regions = np.zeros(pancreas_mask.shape, dtype=np.uint8)
    for i in range(3):
        lo, hi = cuts[i], cuts[i + 1]
        sel = (projection >= lo) & (projection <= hi)
        if sel.any():
            regions[zz[sel], yy[sel], xx[sel]] = i + 1

    return RegionSplit(
        head=(regions == 1),
        body=(regions == 2),
        tail=(regions == 3),
        axis=axis,
        centroid=centroid,
        bounds=tuple(float(c) for c in cuts),
    )


def tumor_location(pancreas_mask, tumor_mask, spacing_xyz) -> str:
    """Name the region holding the tumor's centroid: head, body, tail or unknown."""
    if pancreas_mask is None or tumor_mask is None:
        return "unknown"
    if not pancreas_mask.any() or not tumor_mask.any():
        return "unknown"

    result = _axis_and_projection(pancreas_mask, spacing_xyz)
    if result is None:
        return "unknown"
    _, projection, axis, centroid = result
    cuts = np.linspace(projection.min(), projection.max(), 4)

    tz, ty, tx = np.where(tumor_mask.astype(bool))
    if tx.size == 0:
        return "unknown"
    sx, sy, sz = (float(s) for s in spacing_xyz)
    tumor_coords = np.stack([tx * sx, ty * sy, tz * sz], axis=1)
    centre = ((tumor_coords - centroid) @ axis).mean()

    if centre <= cuts[1]:
        return "head"
    if centre <= cuts[2]:
        return "body"
    return "tail"


def region_overlap(tumor_mask, split: RegionSplit, spacing_xyz) -> dict:
    """How much of the tumor sits in each region, by volume and by share.

    A tumor straddling a boundary gets a share in each region it touches,
    which is more honest than forcing it into the single region holding its
    centroid.

    The three regions tile the *segmented pancreas*, not the whole tumor: a
    lesion growing beyond the organ has voxels in none of them. That remainder
    is reported as ``outside_pancreas`` so the shares always account for the
    whole tumor instead of silently summing to less than 1.
    """
    keys = REGION_NAMES + ("outside_pancreas",)
    if not split.available or tumor_mask is None or not tumor_mask.any():
        return {name: {"volume_cm3": 0.0, "fraction": 0.0} for name in keys}

    sx, sy, sz = (float(s) for s in spacing_xyz)
    voxel_cm3 = (sx * sy * sz) / 1000.0

    tumor = tumor_mask.astype(bool)
    total = float(tumor.sum())

    out = {}
    assigned = 0.0
    for name, mask in split.as_dict().items():
        count = float((tumor & mask).sum())
        assigned += count
        out[name] = {
            "volume_cm3": round(count * voxel_cm3, 3),
            "fraction": round(count / total, 4) if total else 0.0,
        }

    outside = max(0.0, total - assigned)
    out["outside_pancreas"] = {
        "volume_cm3": round(outside * voxel_cm3, 3),
        "fraction": round(outside / total, 4) if total else 0.0,
    }
    return out


def region_volumes(split: RegionSplit, spacing_xyz) -> dict:
    """Volume in cm3 of each pancreas region."""
    if not split.available:
        return {name: 0.0 for name in REGION_NAMES}
    sx, sy, sz = (float(s) for s in spacing_xyz)
    voxel_cm3 = (sx * sy * sz) / 1000.0
    return {
        name: round(float(mask.sum()) * voxel_cm3, 3)
        for name, mask in split.as_dict().items()
    }
