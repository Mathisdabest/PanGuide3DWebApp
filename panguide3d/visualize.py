"""Render overlays of the pancreas regions and the predicted tumor.

Two entry points, for the two things a web app needs:

``overlay_slice``  one axial/coronal/sagittal slice as an RGB array, which is
                   what an interactive scroll-through viewer wants.
``render_views``   three maximum-intensity projections written to PNG, which
                   is what a static report wants.

The colour scheme matches the research repo's qualitative figures so that
app output and paper figures read the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Colours shared with the research repo's qualitative figures.
COLORS = {
    "head": "#1f77b4",
    "body": "#2ca02c",
    "tail": "#9467bd",
    "tumor": "#ff7f0e",
    "tumor_gt": "#8b0000",
    "overlap": "#ff00ff",
}

REGION_ALPHA = 0.35
TUMOR_ALPHA = 0.55


def hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def _blend(base_rgb, mask, color, alpha):
    if mask is None or not mask.any():
        return base_rgb
    rgb = hex_to_rgb(color) if isinstance(color, str) else np.asarray(color, dtype=np.float32)
    sel = mask.astype(bool)
    base_rgb[sel] = (1 - alpha) * base_rgb[sel] + alpha * rgb
    return base_rgb


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to [0,1] and expand to RGB."""
    gray = gray.astype(np.float32)
    lo, hi = float(gray.min()), float(gray.max())
    gray = (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)
    return np.repeat(gray[..., None], 3, axis=2)


def _take_slice(array, view, index):
    if array is None:
        return None
    if view == "axial":
        return array[index]
    if view == "coronal":
        return array[:, index]
    if view == "sagittal":
        return array[:, :, index]
    raise ValueError(f"Unknown view {view!r}; use axial, coronal or sagittal")


def overlay_slice(
    volume_array: np.ndarray,
    index: int,
    view: str = "axial",
    split=None,
    tumor_mask=None,
    show_regions: bool = True,
) -> np.ndarray:
    """One slice with regions and tumor burned in, as float RGB in [0, 1].

    Multiply by 255 and cast to uint8 to hand this to PIL or an HTTP response.
    """
    rgb = _to_rgb(_take_slice(volume_array, view, index))

    if show_regions and split is not None and split.available:
        for name in ("head", "body", "tail"):
            rgb = _blend(rgb, _take_slice(split.as_dict()[name], view, index), COLORS[name], REGION_ALPHA)

    if tumor_mask is not None:
        rgb = _blend(rgb, _take_slice(tumor_mask, view, index), COLORS["tumor"], TUMOR_ALPHA)

    return np.clip(rgb, 0.0, 1.0)


def best_tumor_slice(tumor_mask, view: str = "axial") -> int | None:
    """Index of the slice holding most of the tumor - a good default view."""
    if tumor_mask is None or not tumor_mask.any():
        return None
    axis = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}[view]
    return int(np.argmax(tumor_mask.sum(axis=axis)))


def render_views(
    volume_array: np.ndarray,
    split=None,
    tumor_mask=None,
    out_dir=None,
    case_id: str = "case",
    gt_tumor_mask=None,
):
    """Write axial/coronal/sagittal MIP overlays as PNGs.

    When ``gt_tumor_mask`` is supplied the prediction is drawn against it:
    dark red for ground truth only, orange for prediction only, magenta where
    they agree.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    legend = [("Pancreas head", COLORS["head"]),
              ("Pancreas body", COLORS["body"]),
              ("Pancreas tail", COLORS["tail"])]

    if gt_tumor_mask is not None and tumor_mask is not None:
        overlap = tumor_mask.astype(bool) & gt_tumor_mask.astype(bool)
        layers = [(gt_tumor_mask.astype(bool) & ~overlap, COLORS["tumor_gt"]),
                  (tumor_mask.astype(bool) & ~overlap, COLORS["tumor"]),
                  (overlap, COLORS["overlap"])]
        legend += [("Reference tumor", COLORS["tumor_gt"]),
                   ("Predicted tumor", COLORS["tumor"]),
                   ("Agreement", COLORS["overlap"])]
    elif tumor_mask is not None:
        layers = [(tumor_mask.astype(bool), COLORS["tumor"])]
        legend += [("Predicted tumor", COLORS["tumor"])]
    else:
        layers = []

    written = {}
    for view, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        rgb = _to_rgb(volume_array.max(axis=axis))

        if split is not None and split.available:
            for name in ("head", "body", "tail"):
                rgb = _blend(rgb, split.as_dict()[name].max(axis=axis), COLORS[name], REGION_ALPHA)
        for mask, color in layers:
            rgb = _blend(rgb, mask.max(axis=axis), color, TUMOR_ALPHA)

        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111)
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title(f"{case_id} | {view}")
        ax.axis("off")
        ax.legend(
            handles=[mpatches.Patch(color=hex_to_rgb(c), label=lbl) for lbl, c in legend],
            loc="lower right", fontsize=8, framealpha=0.85,
        )
        path = out_dir / f"{case_id}_{view}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written[view] = path

    return written
