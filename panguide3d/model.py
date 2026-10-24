"""Build the segmentation network and load the shipped checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .models.twostep_nnunet import TwoStepNNUNetTransBottleneck

#: Architecture hyperparameters of the shipped checkpoint. These come from the
#: training run's args.json and must not be changed - the weights won't load.
ARCH = {
    "in_ch": 1,
    "base": 32,
    "levels": 5,
    "max_ch": 320,
    "conditioning": "gating",
    "alpha_pan_gating": 2.0,
}

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "checkpoints" / "panguide3d.pt"


def build_model() -> TwoStepNNUNetTransBottleneck:
    """Instantiate the architecture with the shipped configuration."""
    return TwoStepNNUNetTransBottleneck(**ARCH)


def load_model(weights_path=None, device: str | None = None, strict: bool = True):
    """Build the model and load weights onto ``device``.

    Accepts both the exported weights-only file and a raw training checkpoint
    (which nests the weights under a ``model`` key alongside optimizer state).
    """
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}\n"
            "Weights are not tracked in git - see checkpoints/README.md."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import hashlib
    expected = json.loads(DEFAULT_WEIGHTS.with_suffix(".json").read_text())["sha256"]
    with weights_path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()
    if actual != expected:
        raise ValueError("Checkpoint SHA-256 does not match the inspected model bundle")
    blob = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob

    model = build_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint does not match the architecture: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
            f"First missing: {missing[:3]}, first unexpected: {unexpected[:3]}"
        )

    model.eval().to(device)
    return model, device


def model_info(weights_path=None) -> dict:
    """Read the metadata sidecar written by scripts/export_weights.py."""
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    sidecar = weights_path.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {}
