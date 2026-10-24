"""Strip a training checkpoint down to deployable weights.

Training checkpoints carry optimizer and scheduler state and are roughly
twice the size they need to be. This writes a weights-only file plus a JSON
sidecar recording where it came from.

    python scripts/export_weights.py \
        --checkpoint /lustre/hdd/LAS/qli-lab/xiangm/knopan3d/seg_models/<run>/best_tumor.pt \
        --out checkpoints/panguide3d.pt
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="destination for the weights-only file")
    ap.add_argument("--half", action="store_true",
                    help="store float16 weights, halving the file at a small accuracy cost")
    args = ap.parse_args()

    src = Path(args.checkpoint)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(src, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    train_args = blob.get("args", {}) if isinstance(blob, dict) else {}

    state = {k: (v.half() if args.half and v.is_floating_point() else v)
             for k, v in state.items()}

    torch.save(state, dst)

    n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(src.resolve()),
        "source_run": src.parent.name,
        "model_name": "PanGuide3D",
        "architecture": train_args.get("model", "two_step_nnunet_transbottleneck"),
        "train_args": train_args,
        "parameters": n_params,
        "dtype": "float16" if args.half else "float32",
        "size_mb": round(dst.stat().st_size / 1e6, 1),
        "sha256": sha256(dst),
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] {dst}  ({meta['size_mb']} MB, {n_params/1e6:.1f}M params)")
    print(f"[OK] {dst.with_suffix('.json')}")
    if meta["size_mb"] > 100:
        print("[NOTE] Over GitHub's 100 MB file limit - use Git LFS or host the "
              "weights separately. See checkpoints/README.md.")


if __name__ == "__main__":
    main()
