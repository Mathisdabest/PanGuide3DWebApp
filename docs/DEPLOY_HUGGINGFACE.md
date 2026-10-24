# Deploy without enabling paid hardware

Verified documentation on 2026-10-18: https://huggingface.co/docs/hub/spaces-zerogpu and https://huggingface.co/docs/hub/spaces-overview.

Free personal accounts in good standing (verified email, older than 30 days) may host up to two ZeroGPU Spaces. PRO and organization eligibility differs. Gradio is required. Current default GPU allocation is 48 GB; this app has not been measured there. Anonymous visitors currently receive two GPU minutes/day, free signed-in visitors five, with queue priority depending on identity and remaining quota. No app login is required; provider sign-in may improve quota. Quota, compatibility, and pricing can change. The requested GPU duration is 60 seconds. The first hosted sample inference measured 3.168 seconds (10.526 seconds end to end); larger scans can take longer, and a timeout is reported rather than treated as success. Do not load prepaid credits or enable billable hardware for this project without explicit permission.

1. Create/verify a Hugging Face account. If ineligible for ZeroGPU, keep the local demo while the account ages or request a research/community allocation. A paid subscription or paid GPU is not a free fallback. Exact paid alternatives must be compared at decision time.
2. Run the local tests and build cache: `.venv/bin/python -m pytest -q`, then `.venv/bin/python scripts/build_demo_cache.py MSD_pancreas_045`.
3. Review MODEL_AUDIT.md and public-samples/ATTRIBUTION.md. The checkpoint came from this user's research project. Confirm its authors permit public weight redistribution. A dataset's license alone does not conclusively establish a model weight license. If weight publication is not allowed, put it in a private model repository instead and retrieve at startup using a read-only HF_TOKEN secret and a pinned model revision; never put a token in source. Public access to inference still requires permission from the weight owner.
4. Create a **Gradio Space**, public, and select **ZeroGPU** if offered. Do not select paid hardware. Use Python 3.10 and the SDK version declared in docs/SPACE_README.md. This mirrors the tested local Gradio 4.44.1 API; the hosted Python/ZeroGPU combination must still be validated. The platform may impose newer SDK/security requirements, in which case upgrade and rerun tests before opening access.
5. Prepare a clean folder with `.venv/bin/python scripts/package_space.py`. It excludes all original PanTS files, Git history, credentials, temporary uploads and the local environment. Inspect `space-package/` before upload. Never push the original clone with its PanTS history as the Space.
6. Authenticate locally: `.venv/bin/hf auth login` (paste your write-scoped token into that local prompt, not into chat). Then `.venv/bin/hf upload YOUR_USERNAME/panguide3d space-package . --repo-type space`. The upload includes the checkpoint only if redistribution is permitted. This command uploads files; hardware must already be set to ZeroGPU. No command here enables billing.
7. In Space Settings, set `PANGUIDE_HOST=0.0.0.0`. Gradio Spaces normally choose host automatically, but this application deliberately binds localhost by default for local privacy. Set no telemetry or paid API keys.
8. Wait for build and inspect errors without sharing scan contents. Test cached sample mode first. Then disable cached mode and run the MSD example on ZeroGPU. Measure actual GPU peak/runtime; record them separately from local CPU results. Test anonymous access, quota failure, refresh, concurrent sessions, and TTL cleanup.
9. Open https://huggingface.co/spaces/YOUR_USERNAME/panguide3d in an incognito browser and on a phone. Confirm the iframe provides uploads and PDF downloads; verify Chrome/Safari/Edge separately. A suggested URL is not a deployed URL until tested.
10. Record the demo using docs/DEMO_VIDEO.md. Clearly keep the cached-results label in frame when using recorded outputs.

## Resource controls and limitations

256 MB upload; 512 MB decompression; 40M original and estimated resampled voxels; single-flight inference; queue length eight; three analyses per five minutes per transient salted IP hash. IPs are not logged by app code. Proxy behavior can cause shared limits; distributed abuse prevention would need provider controls. Session data expires after 30 minutes; Gradio files and our report directories are cleaned every minute after a 30-minute age. Active sessions refresh state age. Host/provider logs and backups are outside application guarantees. No claim of secure erasure or HIPAA compliance.

Default CPU fallback works locally and can be slow. Whole-volume arrays and original-grid masks require RAM in addition to the 209 MB model. MONAI keeps stitched logits on CPU and sends only patches to GPU. No browser inference, paid LLM, paid GPU, or patient-data persistence service is used.

Cloud deployment has not been performed without account authentication. A successful local run does not validate ZeroGPU, hosting quotas or browser compatibility.

## Authorized account for this project

The owner supplied username **ilmuuu81** and confirmed ZeroGPU eligibility. After local `hf auth login`, run `.venv/bin/python scripts/deploy_space.py --repo ilmuuu81/panguide3d`. This helper verifies the logged-in owner, refuses to overwrite unrelated Space code, sets only the ZeroGPU hardware API flavor (`zero-a10g`, not paid A10G), sets the public host binding, and uploads the clean package. It stops on authentication/eligibility errors. Verify the actual build and hosted run afterward.
