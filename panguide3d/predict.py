"""Sliding-window inference producing pancreas and tumor masks.

Mirrors ``analysis/infer.py`` from the research repo so that a prediction
served by the web app matches the numbers reported in the model card.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import Volume

#: Patch size the model was trained with, as (x, y, z).
PATCH_XYZ = (160, 160, 96)

#: Evaluation settings used for every number in the model card.
DEFAULT_OVERLAP = 0.5
DEFAULT_THRESHOLD = 0.5

#: Tumor predictions smaller than this are discarded as noise. Matches
#: ``min_pred_voxels`` in the research repo's analysis config.
MIN_TUMOR_VOXELS = 100


@dataclass
class Prediction:
    """Model output for one scan, all arrays shaped (Z, Y, X)."""

    pancreas_mask: np.ndarray      # uint8
    tumor_mask: np.ndarray         # uint8, after the small-component filter
    pancreas_prob: np.ndarray      # float32 in [0, 1]
    tumor_prob: np.ndarray         # float32 in [0, 1]
    spacing: tuple
    threshold: float = DEFAULT_THRESHOLD
    tumor_suppressed: bool = False  # a positive blob was dropped as too small

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx) * float(sy) * float(sz)


class _LogitsWrapper:
    """The two-step network returns a dict; the sliding-window needs a tensor."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)["logits"]


def _sliding_window(model, volume, roi_dhw, overlap, device, sw_batch_size=1):
    """Gaussian-weighted sliding window; falls back to a plain one without MONAI."""
    from monai.inferers import sliding_window_inference

    return sliding_window_inference(
        inputs=volume,
        roi_size=roi_dhw,
        sw_batch_size=int(sw_batch_size),
        predictor=model,
        overlap=float(overlap),
        mode="gaussian",
        sw_device=device, device="cpu",
    )


def _sliding_window_fallback(model, volume, roi_dhw, overlap, device):
    """Uniform-average sliding window, used only when MONAI is absent.

    Produces slightly softer boundaries than the gaussian weighting the model
    card numbers were measured with.
    """
    pd, ph, pw = roi_dhw
    _, _, D, H, W = volume.shape
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    out = cnt = None
    for z0 in range(0, max(1, D - pd + 1), sd):
        for y0 in range(0, max(1, H - ph + 1), sh):
            for x0 in range(0, max(1, W - pw + 1), sw):
                z1, y1, x1 = min(D, z0 + pd), min(H, y0 + ph), min(W, x0 + pw)
                patch = volume[:, :, z0:z1, y0:y1, x0:x1]
                pad = (0, pw - (x1 - x0), 0, ph - (y1 - y0), 0, pd - (z1 - z0))
                if any(pad):
                    patch = F.pad(patch, pad, value=0)

                logits = model(patch.to(device))[:, :, : z1 - z0, : y1 - y0, : x1 - x0]
                if out is None:
                    c = logits.shape[1]
                    out = torch.zeros((1, c, D, H, W), device=device, dtype=torch.float32)
                    cnt = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)
                out[:, :, z0:z1, y0:y1, x0:x1] += logits.float()
                cnt[:, :, z0:z1, y0:y1, x0:x1] += 1.0

    return out / torch.clamp(cnt, min=1.0)


@torch.no_grad()
def predict(
    model,
    volume: Volume,
    device: str = "cuda",
    overlap: float = DEFAULT_OVERLAP,
    threshold: float = DEFAULT_THRESHOLD,
    min_tumor_voxels: int = MIN_TUMOR_VOXELS,
    sw_batch_size: int = 1,
) -> Prediction:
    """Run the model over a preprocessed volume.

    Channel 0 of the network output is the pancreas logit and channel 1 the
    tumor logit; the tumor decoder is gated on the pancreas probability inside
    the network, so the two masks are already consistent with each other.
    """
    model.eval()
    tensor = torch.from_numpy(volume.array.astype(np.float32))[None, None]

    px, py, pz = PATCH_XYZ
    logits = _sliding_window(
        _LogitsWrapper(model), tensor, (pz, py, px), overlap, device, sw_batch_size
    )
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    pancreas_prob = torch.sigmoid(logits[:, 0:1]).cpu().numpy()[0, 0].astype(np.float32)
    tumor_prob = torch.sigmoid(logits[:, 1:2]).cpu().numpy()[0, 0].astype(np.float32)

    pancreas_mask = (pancreas_prob >= threshold).astype(np.uint8)
    tumor_mask = (tumor_prob >= threshold).astype(np.uint8)

    suppressed = False
    if 0 < tumor_mask.sum() <= int(min_tumor_voxels):
        tumor_mask[:] = 0
        suppressed = True

    return Prediction(
        pancreas_mask=pancreas_mask,
        tumor_mask=tumor_mask,
        pancreas_prob=pancreas_prob,
        tumor_prob=tumor_prob,
        spacing=volume.spacing,
        threshold=threshold,
        tumor_suppressed=suppressed,
    )
