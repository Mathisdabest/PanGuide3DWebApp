"""Settle the head/body/tail convention against PanTS's own anatomical labels.

PanTS ships pancreas_head / pancreas_body / pancreas_tail segmentations, so the
true subregion of a lesion is known: it is the subregion its voxels overlap
most. That is the reference here.

Compared against it:
  research    - raw PCA sign, as analysis/stratify.py does it
  patient_right - PCA oriented so the head end is at the patient's right,
                  using each scan's own direction cosines

This reads the raw PanTS tree on the cluster, so it only runs where that data
is mounted; the paths are the constants just below. Last run: patient_right
48/58 (82.8%), research 46/58 (79.3%).

    python scripts/validate_region_split.py 60
"""
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path("/lustre/hdd/LAS/qli-lab/xiangm/knopan3d")
LABELS = ROOT / "data/raw/PanTS/data/LabelTe"
NPZ = ROOT / "data/processed/pants_s1p5_28ch/Te"

REGIONS = ("head", "body", "tail")


def truth_from_anatomy(case):
    """Subregion whose anatomical mask the lesion overlaps most."""
    seg = LABELS / case / "segmentations"
    lesion_path = seg / "pancreatic_lesion.nii.gz"
    if not lesion_path.exists():
        return None
    lesion = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_path))) > 0
    if not lesion.any():
        return None
    overlaps = {}
    for region in REGIONS:
        p = seg / f"pancreas_{region}.nii.gz"
        if not p.exists():
            return None
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
        if mask.shape != lesion.shape:
            return None
        overlaps[region] = int((mask & lesion).sum())
    if sum(overlaps.values()) == 0:
        return None
    return max(overlaps, key=overlaps.get)


def geometric(case, patient_right):
    """Subregion by the equal-thirds PCA split on the preprocessed volume.

    With patient_right=True the axis is oriented using the scan's direction
    cosines so the head end really is on the patient's right; otherwise the
    raw PCA sign is kept, which is what the research code does.
    """
    npz = np.load(NPZ / f"{case}.npz", allow_pickle=True)
    if "label" not in npz.files:
        return None
    lab = npz["label"]
    pan = lab[16] > 0
    tum = lab[27] > 0
    if not pan.any() or not tum.any():
        return None
    spacing = tuple(float(s) for s in npz["spacing"])

    zz, yy, xx = np.where(pan)
    sx, sy, sz = spacing
    coords = np.stack([xx * sx, yy * sy, zz * sz], 1).astype(np.float64)
    mean = coords.mean(0)
    centered = coords - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]

    if patient_right:
        # +x of the stored array points to the patient's right only when the
        # scan's first axis code is R; otherwise it points to the left.
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(LABELS.parent / "ImageTe" / case / "ct.nii.gz"))
        reader.ReadImageInformation()
        code = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            reader.GetDirection()
        )
        x_points_right = code[0] == "R"
        # head is at the patient's right, so head->tail runs toward the left
        want_negative = x_points_right
        if (axis[0] > 0) == want_negative:
            axis = -axis

    projection = centered @ axis
    cuts = np.linspace(projection.min(), projection.max(), 4)
    tz, ty, tx = np.where(tum)
    t = np.stack([tx * sx, ty * sy, tz * sz], 1)
    v = ((t - mean) @ axis).mean()
    return "head" if v <= cuts[1] else ("body" if v <= cuts[2] else "tail")


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cases = []
    seen = set()
    for r in csv.DictReader(open(ROOT / "analysis_out/results_0217/per_case_metrics.csv")):
        if r["model"] != "two_step_nnunet_transbottleneck" or r["cohort"] != "pants":
            continue
        cid = r["case_id"]
        if cid in seen or r["tumor_location"] not in REGIONS:
            continue
        seen.add(cid)
        if (NPZ / f"{cid}.npz").exists() and (LABELS / cid).exists():
            cases.append(cid)
        if len(cases) >= limit:
            break

    tally = {"research": 0, "patient_right": 0}
    n = 0
    rows = []
    for case in cases:
        truth = truth_from_anatomy(case)
        if truth is None:
            continue
        a = geometric(case, patient_right=False)
        b = geometric(case, patient_right=True)
        if a is None or b is None:
            continue
        n += 1
        tally["research"] += a == truth
        tally["patient_right"] += b == truth
        rows.append((case, truth, a, b))
        print(f"{case} truth={truth:5s} research={a:5s} patient_right={b:5s}", flush=True)

    print(f"\n=== against PanTS anatomical labels, n={n} ===", flush=True)
    for k, v in tally.items():
        print(f"  {k:14s} {v}/{n} = {100*v/max(n,1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
