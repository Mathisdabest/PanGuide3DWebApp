"""Turn raw masks into a structured, auditable report for a clinician.

Every number here is measured from the prediction; nothing is invented. The
``reasons`` list states what in the output drove the overall assessment, and
the ``cautions`` list states what would make that assessment unreliable, so a
reader can see the basis for the call rather than only its conclusion.

This is decision *support*. It is not a diagnosis - see LIMITATIONS in the
package README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .predict import Prediction
from .preprocess import Volume
from .regions import (
    REGION_CAVEAT,
    REGION_NAMES,
    RegionSplit,
    region_overlap,
    region_volumes,
    split_pancreas,
    tumor_location,
)

_BENCHMARKS_PATH = Path(__file__).resolve().parent.parent / "reference" / "benchmarks.json"

#: Pancreas volumes outside this range suggest the organ segmentation failed
#: or the scan is not a standard abdominal CT. Normal adult pancreas is
#: roughly 60-120 cm3.
PLAUSIBLE_PANCREAS_CM3 = (25.0, 250.0)

#: Below this mean probability the tumor call rests on weak evidence.
LOW_CONFIDENCE_PROB = 0.70


@dataclass
class Finding:
    """One connected tumor candidate."""

    index: int
    volume_cm3: float
    max_diameter_mm: float
    mean_probability: float
    max_probability: float
    region: str
    region_overlap: dict
    centroid_voxel: tuple


@dataclass
class Report:
    case_id: str
    tumor_detected: bool
    assessment: str
    confidence: str
    pancreas_volume_cm3: float
    region_volumes_cm3: dict
    tumor_volume_cm3: float
    tumor_burden_fraction: float
    dominant_region: str
    findings: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    cautions: list = field(default_factory=list)
    expected_performance: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["findings"] = [asdict(f) if not isinstance(f, dict) else f for f in self.findings]
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _load_benchmarks() -> dict:
    if _BENCHMARKS_PATH.exists():
        return json.loads(_BENCHMARKS_PATH.read_text())
    return {}


def _connected_components(mask):
    """Label 26-connected components, falling back to a single blob."""
    try:
        from scipy.ndimage import label
    except ImportError:
        if mask.any():
            return mask.astype(np.int32), 1
        return np.zeros_like(mask, dtype=np.int32), 0
    return label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))


def _max_diameter_mm(mask_component, spacing_xyz) -> float:
    """Longest extent of a component along any axis, in mm.

    An axis-aligned extent rather than a true caliper measurement; it is
    cheap and does not understate the lesion.
    """
    zz, yy, xx = np.where(mask_component)
    if xx.size == 0:
        return 0.0
    sx, sy, sz = (float(s) for s in spacing_xyz)
    return float(max(
        (xx.max() - xx.min() + 1) * sx,
        (yy.max() - yy.min() + 1) * sy,
        (zz.max() - zz.min() + 1) * sz,
    ))


def build_findings(prediction: Prediction, split: RegionSplit) -> list:
    """One Finding per connected tumor component, largest first."""
    labelled, n = _connected_components(prediction.tumor_mask)
    if n == 0:
        return []

    voxel_cm3 = prediction.voxel_volume_mm3 / 1000.0
    findings = []
    for i in range(1, n + 1):
        component = labelled == i
        count = int(component.sum())
        if count == 0:
            continue
        probs = prediction.tumor_prob[component]
        zz, yy, xx = np.where(component)
        findings.append(Finding(
            index=i,
            volume_cm3=round(count * voxel_cm3, 3),
            max_diameter_mm=round(_max_diameter_mm(component, prediction.spacing), 1),
            mean_probability=round(float(probs.mean()), 4),
            max_probability=round(float(probs.max()), 4),
            region=tumor_location(prediction.pancreas_mask, component, prediction.spacing),
            region_overlap=region_overlap(component, split, prediction.spacing),
            centroid_voxel=(int(zz.mean()), int(yy.mean()), int(xx.mean())),
        ))

    findings.sort(key=lambda f: -f.volume_cm3)
    for rank, finding in enumerate(findings, start=1):
        finding.index = rank
    return findings


def _confidence(findings, pancreas_cm3, prediction) -> str:
    """Coarse confidence band for the tumor call."""
    if not findings:
        return "low" if pancreas_cm3 <= 0 else "moderate"
    top = findings[0]
    if top.mean_probability >= 0.85 and top.volume_cm3 >= 1.0:
        return "high"
    if top.mean_probability >= LOW_CONFIDENCE_PROB:
        return "moderate"
    return "low"


def build_report(
    volume: Volume,
    prediction: Prediction,
    split: RegionSplit | None = None,
    benchmarks: dict | None = None,
) -> Report:
    """Assemble the full structured report for one scan."""
    if split is None:
        split = split_pancreas(prediction.pancreas_mask, prediction.spacing)
    if benchmarks is None:
        benchmarks = _load_benchmarks()

    voxel_cm3 = prediction.voxel_volume_mm3 / 1000.0
    pancreas_cm3 = round(float(prediction.pancreas_mask.sum()) * voxel_cm3, 2)
    tumor_cm3 = round(float(prediction.tumor_mask.sum()) * voxel_cm3, 3)

    findings = build_findings(prediction, split)
    detected = bool(findings)

    overall_overlap = region_overlap(prediction.tumor_mask, split, prediction.spacing)
    # Only the three anatomical regions can be dominant; the "outside_pancreas"
    # remainder is reported separately, never named as a location.
    dominant = (
        max(REGION_NAMES, key=lambda r: overall_overlap[r]["fraction"])
        if detected and split.available
        else "n/a"
    )

    region_cm3 = region_volumes(split, prediction.spacing)
    reasons, cautions = [], []

    # --- what the model found -------------------------------------------
    if pancreas_cm3 > 0:
        if split.available:
            reasons.append(
                f"Pancreas segmented at {pancreas_cm3} cm3, split into "
                f"head {region_cm3['head']} cm3, body {region_cm3['body']} cm3, "
                f"tail {region_cm3['tail']} cm3."
            )
        else:
            reasons.append(f"Pancreas segmented at {pancreas_cm3} cm3.")
    else:
        reasons.append("No pancreas was segmented in this volume.")
        cautions.append(
            "Without a pancreas segmentation the tumor channel is unreliable: the "
            "tumor decoder is gated on the pancreas probability."
        )

    if detected:
        top = findings[0]
        reasons.append(
            f"{len(findings)} tumor candidate{'s' if len(findings) > 1 else ''} above "
            f"probability {prediction.threshold}; largest is {top.volume_cm3} cm3 "
            f"({top.max_diameter_mm} mm across) in the {top.region}."
        )
        reasons.append(
            f"Mean predicted tumor probability in the largest candidate is "
            f"{top.mean_probability} (peak {top.max_probability})."
        )
        if split.available:
            share = overall_overlap[dominant]["fraction"]
            reasons.append(
                f"{share:.0%} of the predicted tumor volume lies in the pancreatic {dominant}."
            )
        if len(findings) > 1:
            cautions.append(
                f"{len(findings)} separate components were predicted. Multifocal disease is "
                "uncommon; the smaller components may be false positives."
            )
        if top.mean_probability < LOW_CONFIDENCE_PROB:
            cautions.append(
                f"Largest candidate has a mean probability of only {top.mean_probability}, "
                "close to the decision threshold."
            )
    else:
        reasons.append(
            f"No voxel cluster exceeded probability {prediction.threshold} in the tumor channel."
        )
        peak = float(prediction.tumor_prob.max()) if prediction.tumor_prob.size else 0.0
        reasons.append(f"Peak tumor probability anywhere in the volume was {peak:.3f}.")
        if prediction.tumor_suppressed:
            cautions.append(
                "A positive cluster was found but discarded for being under the "
                "100-voxel minimum size. A small lesion may have been removed."
            )
        elif peak >= 0.35:
            cautions.append(
                f"Peak tumor probability of {peak:.3f} is close to the threshold; a "
                "sub-threshold lesion cannot be excluded."
            )

    # --- what would make this unreliable --------------------------------
    if 0 < pancreas_cm3 < PLAUSIBLE_PANCREAS_CM3[0]:
        cautions.append(
            f"Segmented pancreas volume ({pancreas_cm3} cm3) is below the plausible "
            "range for an adult; the organ may be partly outside the field of view."
        )
    elif pancreas_cm3 > PLAUSIBLE_PANCREAS_CM3[1]:
        cautions.append(
            f"Segmented pancreas volume ({pancreas_cm3} cm3) is above the plausible "
            "range for an adult; the segmentation has likely leaked into adjacent tissue."
        )

    if split.available:
        cautions.append(REGION_CAVEAT)

    cautions.append(
        "Model trained and validated on contrast-enhanced abdominal CT only. "
        "It has not been evaluated on MRI and must not be used on MR studies."
    )

    # --- how well the model does overall ---------------------------------
    # Only cohort-level numbers are quoted, because those are the only ones the
    # PanGuide3D paper tabulates. It analyses performance by subregion and tumor
    # size in a figure, but publishes no numbers for those strata, so the report
    # does not claim any.
    expected = dict(benchmarks.get("overall", {}))
    if expected:
        reasons.append(
            f"On the published in-cohort test set PanGuide3D reached a tumor Dice of "
            f"{expected.get('tumor_dice', 'n/a')} and a tumor sensitivity of "
            f"{expected.get('tumor_sensitivity', 'n/a')}."
        )
    if detected:
        cautions.append(
            "Published performance is reported per cohort, not per pancreatic subregion. "
            "No subregion-specific accuracy can be quoted for this finding."
        )

    assessment = (
        f"Tumor suspected in the pancreatic {dominant}" if detected
        else "No pancreatic tumor detected"
    )

    return Report(
        case_id=volume.case_id,
        tumor_detected=detected,
        assessment=assessment,
        confidence=_confidence(findings, pancreas_cm3, prediction),
        pancreas_volume_cm3=pancreas_cm3,
        region_volumes_cm3=region_cm3,
        tumor_volume_cm3=tumor_cm3,
        tumor_burden_fraction=round(tumor_cm3 / pancreas_cm3, 4) if pancreas_cm3 else 0.0,
        dominant_region=dominant,
        findings=findings,
        reasons=reasons,
        cautions=cautions,
        expected_performance=expected,
        meta={
            "model": benchmarks.get("model_name", "PanGuide3D"),
            "threshold": prediction.threshold,
            "spacing_mm": list(prediction.spacing),
            "original_spacing_mm": list(volume.original_spacing),
            "original_size": list(volume.original_size),
            "volume_shape_zyx": list(volume.shape),
        },
    )
