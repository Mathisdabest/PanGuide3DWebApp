# Weights

The file `panguide3d.pt` (209 MB) is **not tracked in git** —
it is over GitHub's 100 MB per-file limit. `panguide3d.json`
is tracked and records exactly which checkpoint it came from, its SHA-256, and
the full training configuration.

| | |
|---|---|
| Architecture | PanGuide3D (internal training name: `two_step_nnunet_transbottleneck`) |
| Parameters | 52.3 M |
| Size | 209.3 MB (float32) |
| Source | `seg_models/two_step_nnunet_transbottleneck_multilabel_R2_patch160x160x96_msd_cap320_nnunetstyle_bs_2_accum_1_AdvAug_lr1e-2_full_0212/best_pos.pt` |
| Epoch | 714 of 1000 |

## Getting the file

**On the cluster** — copy it from the research repo:

```bash
cp /lustre/hdd/LAS/qli-lab/xiangm/knopan3d/panguide3d_webapp/checkpoints/panguide3d.pt \
   checkpoints/
```

**Re-export it from the training checkpoint:**

```bash
python scripts/export_weights.py \
  --checkpoint /lustre/hdd/LAS/qli-lab/xiangm/knopan3d/seg_models/two_step_nnunet_transbottleneck_multilabel_R2_patch160x160x96_msd_cap320_nnunetstyle_bs_2_accum_1_AdvAug_lr1e-2_full_0212/best_pos.pt \
  --out checkpoints/panguide3d.pt
```

Add `--half` for a ~105 MB float16 file. Still over the GitHub limit, and it
costs a little accuracy, so prefer float32 unless deployment size is binding.

## Distributing it

**Git LFS** keeps the weights in the repo:

```bash
git lfs install
git lfs track "checkpoints/*.pt"
git add .gitattributes
# then drop the checkpoints/*.pt line from .gitignore
```

Mind the quota — LFS storage on free GitHub accounts is 1 GB.

**Or host it separately** (a release asset, S3, Hugging Face) and fetch it at
deploy time. For a containerized app this is usually simpler: bake the weights
into the image rather than pulling them at runtime.

## Verify

```bash
sha256sum checkpoints/panguide3d.pt
# compare against "sha256" in panguide3d.json
```
