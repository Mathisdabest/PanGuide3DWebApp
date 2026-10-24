# Sharing this folder, and hosting the model

Written for whoever takes this repo over and builds the web app.

---

## 1. Licensing — settle this before anything is public

**Decided for this project:** the code is MIT (`LICENSE`), and the PanTS demo
scans stay as they are, under a **non-commercial, educational-use** reading,
with the dataset cited in `samples/README.md`. The analysis below is kept
because it still matters if this is ever monetised — at that point the NC
clause bites and the demo has to move to MSD.

Three different licences apply to three different things here.

| Thing | Origin | Licence | Can it go in a public repo? |
|---|---|---|---|
| Code in `panguide3d/`, `scripts/` | this project | **MIT** — see `LICENSE` | yes |
| `checkpoints/panguide3d.pt` | trained on MSD Task07 | MSD is **CC BY-SA 4.0** | yes, with attribution |
| `samples/*/input/*` | **PanTS** | **CC BY-NC-ND 4.0** | **see below** |

**The weights are fine.** The shipped checkpoint was trained on MSD Task07
only — its `args.json` records `data_dir: data/processed/msd_pancreas_s1p5_28ch`
— and MSD is CC BY-SA 4.0, which permits redistribution and derivatives with
attribution and share-alike.

**The bundled demo scans are the problem.** They come from PanTS, which is
CC BY-**NC**-**ND** 4.0:

- **NC — NonCommercial.** A free public demo is fine. Anything monetised is not.
- **ND — NoDerivatives.** This is the binding one. The `.npz` files are
  *modified* copies of PanTS volumes — resampled to 1.5 mm, HU-windowed,
  relabelled into a 28-channel array — so redistributing them is a
  redistribution of derivative works, which ND does not permit. The
  `ct.nii.gz` files are byte-identical copies, so those are redistributable
  with attribution for non-commercial use. The overlay PNGs are also derived
  renderings and sit in the same grey area as the `.npz`.

PanTS also notes "Paper content is covered by patents pending."

### Recommended fix: rebuild the demo on MSD

MSD Task07 is CC BY-SA 4.0 — redistribution *and* derivatives both allowed —
and it is the data this checkpoint was trained on. Swapping the demo cases
removes the whole problem. Good candidates from the held-out fold-0 split, with
the Dice this checkpoint achieved:

| Case | Shows | Tumor Dice | Raw size |
|---|---|---|---|
| `pancreas_137` | head | 0.865 | 26 MB |
| `pancreas_256` | body | 0.801 | 25 MB |
| `pancreas_045` | tail | 0.801 | 12 MB |

One catch: **MSD has no tumor-free cases** — all 57 held-out cases are
tumor-positive — so a negative example has to come from PanTS or be dropped.
Options, in order of how clean they are:

1. Ship three MSD cases, no negative. Fully CC BY-SA, no restrictions carried.
2. Ship three MSD cases plus the *verbatim* PanTS `ct.nii.gz` for the negative,
   with attribution. Adds an NC restriction to the repo.
3. Keep the current PanTS set. Simplest, but the repo inherits NC-ND and the
   `.npz` files are hard to defend.

Whatever you choose, include the citations for the datasets you ship — PanTS is
`arXiv:2507.01291`, MSD Task07 is Memorial Sloan Kettering via the Medical
Segmentation Decathlon — and cite the model paper, `arXiv:2604.20981`.

Also note `samples/README.md` currently describes the PanTS set; update it if
you swap the cases.

---

## 2. Handing the folder over, step by step

The repo is ~113 MB of tracked files plus a 200 MB checkpoint that is
deliberately *not* in git. So two things travel: a **git bundle** (the whole
repo and its history in one file) and the **weights**.

These commands were run end to end and work as written.

### Step 1 — licence (done)

`LICENSE` is MIT, which lets her build on, relicense and publish the code
freely. It covers the code only; the weights and the demo scans carry their own
dataset terms, and the file says so. The dataset question in §1 is settled:
PanTS scans stay, cited, non-commercial.

### Step 2 — make the repo and the bundle (on the cluster)

```bash
cd /lustre/hdd/LAS/qli-lab/xiangm/knopan3d/panguide3d_webapp

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
git init -b main
git add -A
git commit -m "PanGuide3D web app package"

git bundle create panguide3d.bundle --all
```

That produces a single ~113 MB file next to the repo. `.gitignore` excludes
`*.bundle`, so it never gets committed into the repo it contains. Keep it on
Lustre, not in `~` — `/home` here is quota-limited and already at 2.2 GB. `.gitignore` keeps the checkpoint out of
it, which is what you want — it travels separately.

### Step 3 — send her two files

| File | Size | How |
|---|---|---|
| `panguide3d.bundle` | 113 MB | Google Drive, Box, OneDrive, `scp` — anything |
| `checkpoints/panguide3d.pt` | 200 MB | same |

Send the SHA-256 from `checkpoints/panguide3d.json` too, so she can verify the
weights arrived intact.

### Step 4 — she restores it locally

```bash
git clone panguide3d.bundle panguide3d-webapp
cd panguide3d-webapp

mkdir -p checkpoints
cp /path/to/panguide3d.pt checkpoints/
sha256sum checkpoints/panguide3d.pt        # compare with checkpoints/panguide3d.json

pip install -r requirements.txt
python scripts/make_examples.py            # full self-test, ~3 min on a laptop CPU
```

If that last command reproduces the four reports and overlays, everything
arrived intact. The clone carries full history, so she can keep committing
normally from there.

### Step 5 — she pushes to her own GitHub

She creates an empty repository on her account — **no** README, licence or
.gitignore, so there is nothing to merge — then:

```bash
git remote remove origin                   # points at the bundle file
git remote add origin https://github.com/HER-USERNAME/panguide3d-webapp.git
git push -u origin main
```

She owns it outright from that point; nothing refers back to your account.

### Step 6 — publish the weights

The 200 MB checkpoint still cannot go in plain git. On her new repo:

**GitHub Release** (simplest):
*Releases → Draft a new release → tag `v0.1.0` → attach `panguide3d.pt` → Publish.*
2 GB per-file limit, free, no LFS quota.

**Or a Hugging Face model repo** — better if she is going the ZeroGPU route,
since the Space can then pull weights from the Hub directly:

```bash
huggingface-cli upload HER-USERNAME/panguide3d checkpoints/panguide3d.pt
```

Then add a fetch script so a fresh clone is one command from running:

```bash
cat > scripts/fetch_weights.sh <<'SH'
#!/usr/bin/env bash
set -e
mkdir -p checkpoints
curl -L -o checkpoints/panguide3d.pt \
  https://github.com/HER-USERNAME/panguide3d-webapp/releases/download/v0.1.0/panguide3d.pt
sha256sum -c <<<"$(python -c "import json;print(json.load(open('checkpoints/panguide3d.json'))['sha256'])")  checkpoints/panguide3d.pt"
SH
chmod +x scripts/fetch_weights.sh
```

### Why a bundle rather than just pushing it yourself

A bundle makes her the sole owner from the first commit — no fork relationship,
no transfer request, no trace of your account in the history's remote config.
If you would rather she just forked yours, push it to your GitHub and have her
fork; it works, but the fork stays visibly linked to your repo.

## 3. Hosting the model

### Measured cost, CPU vs GPU

Same scan, same checkpoint, both measured:

| | 2 CPU cores | A100 GPU |
|---|---|---|
| Weight load | 1.0 s | 1.5 s |
| Preprocessing | 2.6 s | 1.3 s |
| **Inference** | **221 s** | **0.67 s** |
| Memory | 3.6 GB RAM | 2.85 GB VRAM |

Inference is **330× faster on a GPU** — about two seconds end to end instead of
nearly four minutes. That single fact decides the architecture.

Two things the CPU numbers also settle:

- **Serverless is out.** Vercel, Netlify, Cloudflare Workers and Lambda cap at
  10–60 s and 1–3 GB. CPU inference needs 3.6 GB and nearly four minutes. This
  is an architecture mismatch, not a tier you can pay your way out of.
- **`overlap` is not a speed knob.** 0.25 and 0.50 measured 209.4 s and 209.8 s
  — identical. A 1.5 mm volume tiles into the same patch count either way, so
  halving the overlap halves nothing. Don't trade accuracy for it.

One GPU caveat: the **first** inference in a fresh process took 37 s while
cuDNN autotuned kernels for these 3D convolution shapes; every subsequent call
was 0.67 s. Budget for that on a cold start, or do a throwaway warm-up pass at
startup.

### Recommended: Hugging Face ZeroGPU

[ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) attaches a GPU
only while a request is running, and it is genuinely free:

| | |
|---|---|
| Hardware | half an RTX Pro 6000 Blackwell, 48 GB VRAM (default `large`) |
| Hosting | **free personal accounts: up to 2 ZeroGPU Spaces** |
| Account requirements | verified email, **account older than 30 days** |
| SDK | **Gradio only** — no Docker, no custom server |
| Per-call limit | 60 s default, raise with `@spaces.GPU(duration=...)` |
| Visitor quota | unauthenticated 2 min/day, free account 5 min/day, PRO 40 min/day |

This model needs 2.85 GB of the 48 GB available and ~2 s per scan, so it fits
with enormous headroom. The daily quota is charged to **the visitor**, not the
Space owner, so a public demo does not burn your allowance: at ~2 s a scan, even
an anonymous visitor's 2-minute allowance is roughly 60 analyses a day.

Three constraints worth planning around:

1. **The 30-day account age is a hard gate.** If your friend's Hugging Face
   account is new, she cannot host a ZeroGPU Space yet. Create the account
   now so the clock starts.
2. **ZeroGPU is Gradio-only.** If the web app is to be a custom React/Svelte
   design, it cannot *be* the Space. Build it as a separate static site and
   call the Space as a backend (below).
3. Load the model onto CUDA at **module level**, not inside the decorated
   function — the docs are explicit that lazy-loading into `@spaces.GPU` is
   much slower.

Sketch:

```python
import spaces, gradio as gr
from panguide3d import PanGuide3D

analyzer = PanGuide3D(device="cuda")        # module level, as ZeroGPU requires

@spaces.GPU(duration=120)                   # 120s absorbs the cold-start autotune
def analyze(path):
    result = analyzer.analyze(path)
    return result.report.to_dict()

gr.Interface(fn=analyze, inputs=gr.File(), outputs=gr.JSON()).launch()
```

**The free CPU tier is no longer an option, and this is now explicit in the
docs.** The Spaces overview states: "Gradio and Docker Spaces run on compute
and require a paid plan to create: PRO for personal accounts, Team or
Enterprise for organizations. Free personal accounts in good standing can still
host up to 2 Gradio Spaces running on ZeroGPU."

So the free paths are exactly two: a **static** Space (no backend, no
inference) or a **ZeroGPU** Gradio Space. CPU Basic hardware is still listed at
no hourly cost, but a free personal account cannot create the Gradio Space that
would run on it. Tolerating a 4-minute CPU inference therefore does not buy a
free deployment — it only describes the fallback if she ever runs the app on
her own machine or on paid CPU hardware ($0.03/h CPU Upgrade, or PRO at
$9/month to create a CPU Basic Gradio Space).

### Live upload works, and the path is already proven

Nothing about hosting rules out upload — `PanGuide3D.analyze()` takes a file
path, so a Gradio `gr.File` input is the whole integration. The raw-NIfTI route
is already validated end to end: every case in `samples/` was run twice, once
from the preprocessed `.npz` and once from the original `ct.nii.gz`, and the
reports agree to the third decimal (`samples/index.json`, the
`from_preprocessed_npz` and `from_raw_nifti` blocks). An uploaded scan goes
through exactly the same code as the second of those. Uploads are 14–42 MB for
the demo cases, which is unremarkable for Gradio.

Four things to wire up, none of them large:

1. **`duration` must cover the cold start.** The first inference in a fresh
   process spends ~37 s on cuDNN autotuning; later calls are 0.67 s.
   `@spaces.GPU(duration=120)` covers it. Do not inflate it beyond that —
   ZeroGPU charges the visitor's daily quota against the *requested* duration,
   not the actual runtime, so an oversized `duration` makes visitors with
   little quota left fail outright.
2. **Reject non-CT before inference.** `preprocess.looks_like_ct()` already
   exists; on MR the model returns a confident wrong answer rather than an
   error.
3. **Do not persist uploads.** Set `flagging_mode="never"` (Gradio's flagging
   writes uploads to disk), delete the temp file in a `finally:`, and say so on
   the page. Spaces disk is 50 GB and non-persistent, but "not persistent" is
   not the same as "never written".
4. **Keep `samples/` as the landing view.** Precomputed results render with no
   GPU, no quota and no cold start, so the page still works when the Space is
   asleep or a visitor is out of quota. Upload becomes the "try your own scan"
   path rather than the only path.

Quota arithmetic for a public demo, charged to **the visitor**, not to her: an
anonymous visitor gets 2 min/day, a free account 5 min/day, PRO 40 min/day. At
a ~2 s scan plus queueing, even the anonymous allowance is tens of analyses a
day.

### If you want a custom-designed front end

Host the UI wherever you like — GitHub Pages, Vercel, Netlify, all free for a
static site — and call the Gradio Space from it:

```js
import { Client } from "@gradio/client";
const app = await Client.connect("her-username/panguide3d");
const out = await app.predict("/analyze", [file]);
```

The Space becomes a pure inference API; the Space's own Gradio UI can stay as
an internal fallback. This is the setup I would pick: full design control on
the front end, free GPU on the back.

### Still worth precomputing the four demo cases

Whatever you build, keep the bundled results in `samples/` as the landing
experience. They render instantly, need no GPU, survive quota exhaustion and
cold starts, and work if the Space is sleeping. Live upload is then the
"try it yourself" path rather than the only path.

## 4. Before this faces the public

This outputs cancer predictions, so a few things are not optional.

- **Say what it is, on the page, not buried in a README.** Research
  demonstration, not a medical device, not cleared by any regulator, not a
  diagnosis. `report.cautions` already carries per-case wording — surface it.
- **Do not accept real patient scans.** If anyone can upload, you are handling
  health data, with the privacy and regulatory weight that implies. The
  precomputed design avoids this entirely. If you do allow upload: state that
  files are processed in memory and never stored, and then actually do that.
- **CT only.** The model cannot process MRI — its preprocessing is a
  Hounsfield-unit window, which MR does not produce. It will return a confident
  wrong answer rather than an error. `preprocess.looks_like_ct()` exists for
  this; wire it to a rejection path with a clear message.
- **Show the error rates next to the result.** Tumor sensitivity is 0.787 —
  roughly one in five missed — and false positives average 3 cm³ per case. A
  clean-looking negative is not an all-clear, and a flagged region is not a
  diagnosis. `report.expected_performance` has the published numbers.
- **Head/body/tail is approximate.** The split is geometric and matches PanTS's
  own anatomical labels about 83% of the time. `regions.REGION_CAVEAT` has
  display-ready wording.

---

## 5. Suggested order of work

1. ~~Add a `LICENSE` and settle the dataset question in §1.~~ **Done** — MIT,
   PanTS scans kept and cited, non-commercial use only.
2. **Create her Hugging Face account today** if she does not have one —
   ZeroGPU hosting needs it to be 30 days old, so the clock should start now
   even if the Space comes later.
3. **Hand over** the bundle and the weights (§2), and confirm
   `python scripts/make_examples.py` reproduces on her machine.
4. **Push to her GitHub**, weights as a Release asset, add `fetch_weights.sh`.
5. **Build the static viewer** over `samples/` — `index.json`, the overlay
   PNGs, the reports. Deploy to GitHub Pages or Vercel. This is a complete,
   shareable demo with no backend at all.
6. **Add the ZeroGPU Space** for live upload, and point the front end at it.

Steps 1–5 need no GPU and no running server. Step 6 is what turns it into a
live tool, and by then the account will be old enough to host it.
