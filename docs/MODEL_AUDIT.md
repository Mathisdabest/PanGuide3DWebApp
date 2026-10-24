# Inspection of the actual bundle

Original inputs are preserved at ../panguide3d.bundle and ../panguide3d.pt. This app is an isolated clone of bundle commit 9cdb8dc85a189f8272b2ae4a865a515510c73b83.

Checkpoint SHA-256 verified locally: 917acfa357740361e8d25bbe4287c4e861e67d685ad95ff745b82a41f154a1b0. ZIP/pickle opcodes inspected without execution: OrderedDict, torch FloatStorage, torch._utils _rebuild_tensor_v2. Loader now verifies the digest and uses weights_only=True. It does not allow arbitrary uploaded checkpoints.

Architecture: PyTorch TwoStepNNUNetTransBottleneck, 52,328,834 parameters, 209,345,333 bytes. Shared encoder, two decoders, probabilistic pancreas gating (alpha 2), two-block Transformer bottleneck. Input B×1×Z×Y×X float32. Outputs dict containing logits B×2×Z×Y×X, channel 0 pancreas and channel 1 tumor. Independent sigmoid with 0.5 threshold, not a mutually exclusive softmax. Masks can overlap. There is no cancer classification head.

Actual metadata: checkpoints/panguide3d.json records MSD Task07, fold 0, patch_xyz=[160,160,96], base=32, levels=5, cap=320, gating. This conflicts with general PanTS training statements in the inherited README/model card. Those benchmark tables are not asserted to measure this checkpoint. No complete training split, training log, raw training preprocessing script, or independent evaluation was supplied.

Bundled preprocessing: SimpleITK RAS orientation, 1.5 mm isotropic B-spline, clip [-125,275] HU, scale [0,1]. Gaussian MONAI sliding windows, overlap .5, batch 1, ROI ZYX=(96,160,160). The inherited code suppresses the *entire tumor mask* when 0<total voxels<=100; it does not remove each small connected component. This behavior is preserved for fidelity and stated in reports. Uniform fallback inference is not used in the application requirements.

Orientation limitation: original notes say training was mixed orientation without canonicalization, while bundled inference canonicalizes to RAS. This can change outputs for mirrored inputs. The absent training script prevents a stronger claim of exact training equivalence. No arbitrary alternative preprocessing has been introduced. Public UI rejects oblique inputs; axis-aligned masks are mapped using origin/direction/spacing to the original scan grid by nearest-neighbor resampling, then displayed using lossless LPS permutation/flips. CT/MR cannot be reliably distinguished from NIfTI alone: the HU-range gate is a heuristic. Contrast phase is user responsibility.

DICOM: one CT series only, constant orientation, dimensions and spacing, spatial sorting, rejection of duplicate/missing/irregular positions, gantry tilt and enhanced multiframe, GDCM intensity rescale. Common identifier fields are read during parsing but not copied into output volumes or reports; raw files remain temporary. BurnedInAnnotation=YES is rejected. Other burned-in text cannot be detected reliably. No HIPAA claim.

Report: deterministic experimental segmentation statistics, six-connected component count, physical mask volumes, intersection volume, percentage of tumor inside pancreas and pancreas intersected. No confidence label or diagnosis. Head/body/tail approximations from the inherited pipeline are not presented as model classes.

Performance: generated demo-cache/*.json contains actual local device, shapes, inference duration and process high-water RSS. RSS is a process lifetime peak, not isolated inference allocation; on macOS resource reports bytes. The completed Apple MPS inference took 135.63 seconds. CUDA peak/runtime cannot be measured on this Mac; MPS unified GPU peak was not recorded. Original deployment notes contain prior cluster figures, not measurements of this deployment. ZeroGPU compatibility remains provisional until a hosted run.

Resolved during implementation: owner authentication, eligible ZeroGPU account, public deployment, and one measured cloud inference. Remaining provenance gaps: independent validation of the training split and orientation recipe, and a separate explicit weight-license document beyond the source project’s stated dataset terms. Public-domain or permissively licensed CT replacement is supplied under public-samples; original PanTS derivatives are excluded from packaging.

## Hosted verification

The Space is public at https://huggingface.co/spaces/ilmuuu81/panguide3d, with ZeroGPU hardware confirmed. Anonymous sample inference succeeded; a provider-authenticated browser upload succeeded with 3.079 seconds measured model runtime and 2.824 GiB peak CUDA allocation (7.334 seconds end to end). This is one selected case and not a general speed or accuracy guarantee. An additional anonymous upload test timed out; quota/queue behavior remains a hosting constraint. See docs/VALIDATION.md and deliverables/public-verification.json.
