"""Generate the bundled demo cases in samples/.

Runs PanGuide3D over four hand-picked PanTS test cases - one tumor in each
pancreatic subregion, plus a tumor-free case - and writes the report JSON and
the overlay renders a web app would display.

Each case is run twice, once from the raw ``ct.nii.gz`` and once from the
research ``.npz``, to show both input paths work and that this package's
preprocessing reproduces the research pipeline.

These cases were picked to be legible demonstrations, so they are *not* a
performance estimate - MODEL_CARD.md has the measured numbers.

The scans themselves are bundled under ``samples/<case>/input/``, so this runs
with no external data:

    python scripts/make_examples.py

Pass ``--data_root <research repo>`` to read the cluster copies instead.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panguide3d import PanGuide3D                       # noqa: E402
from panguide3d.preprocess import Volume                # noqa: E402
from panguide3d.regions import tumor_location           # noqa: E402
from panguide3d.visualize import render_views           # noqa: E402

PANCREAS_CHANNEL, TUMOR_CHANNEL = 16, 27

# case_id, what the case is meant to show.
#
# All four are PanTS test cases, genuinely unseen by the shipped checkpoint.
# The three tumor cases were picked as clear examples of each subregion. The
# negative was picked by running the shipped weights over tumor-free candidates
# and keeping one the model calls correctly - most produce a small false
# positive, which is what the published false-positive volume describes. All
# four are R-first scans, so reorientation is a no-op for them and the raw and
# preprocessed paths are directly comparable.
CASES = [
    ("PanTS_00009473", "tumor in the pancreatic head"),
    ("PanTS_00009005", "tumor in the pancreatic body"),
    ("PanTS_00009001", "tumor in the pancreatic tail"),
    ("PanTS_00009160", "no tumor (negative case)"),
]


def load_npz(path: Path):
    """Read an already-preprocessed research .npz plus its reference labels."""
    npz = np.load(path, allow_pickle=True)
    spacing = tuple(float(s) for s in npz["spacing"])
    volume = Volume(
        array=npz["image"].astype(np.float32),
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        direction=tuple(),
        original_spacing=spacing,
        original_size=tuple(int(v) for v in npz["image"].shape[::-1]),
        case_id=path.stem,
    )
    reference = None
    if "label" in npz.files:
        labels = npz["label"]
        reference = {
            "pancreas": (labels[PANCREAS_CHANNEL] > 0).astype(np.uint8),
            "tumor": (labels[TUMOR_CHANNEL] > 0).astype(np.uint8),
        }
    return volume, reference


def dice(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)


def summarize(report, elapsed) -> dict:
    return {
        "tumor_detected": report.tumor_detected,
        "assessment": report.assessment,
        "confidence": report.confidence,
        "dominant_region": report.dominant_region,
        "tumor_volume_cm3": report.tumor_volume_cm3,
        "pancreas_volume_cm3": report.pancreas_volume_cm3,
        "region_volumes_cm3": report.region_volumes_cm3,
        "runtime_seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default="",
                    help="research repo root; only needed for cases whose scans "
                         "are not bundled under samples/<case>/input/")
    ap.add_argument("--out_dir", default="samples")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip_raw", action="store_true",
                    help="only run the preprocessed .npz path")
    ap.add_argument("--only", default="",
                    help="comma-separated case ids to run, instead of all of them")
    args = ap.parse_args()

    wanted = {c.strip() for c in args.only.split(",") if c.strip()}

    root = Path(args.data_root) if args.data_root else None
    npz_dir = root / "data" / "processed" / "pants_s1p5_28ch" / "Te" if root else None
    raw_dir = root / "data" / "raw" / "PanTS" / "data" / "ImageTe" if root else None
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def inputs_for(case_id):
        """Bundled copies first, so this runs with no external data."""
        bundled = out_root / case_id / "input"
        npz = bundled / f"{case_id}.npz"
        raw = bundled / "ct.nii.gz"
        if not npz.exists() and npz_dir is not None:
            npz = npz_dir / f"{case_id}.npz"
        if not raw.exists() and raw_dir is not None:
            raw = raw_dir / case_id / "ct.nii.gz"
        return npz, raw

    analyzer = PanGuide3D(weights_path=args.weights, device=args.device)
    print(f"[model] PanGuide3D on {analyzer.device}", flush=True)

    index = []
    for case_id, intent in CASES:
        if wanted and case_id not in wanted:
            continue
        npz_path, raw_path = inputs_for(case_id)
        if not npz_path.exists():
            print(f"[skip] {case_id}: {npz_path} not found", flush=True)
            continue

        print(f"\n[case] {case_id} - {intent}", flush=True)
        case_dir = out_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # --- preprocessed .npz path --------------------------------------
        volume, reference = load_npz(npz_path)
        started = time.time()
        result = analyzer._run(volume)
        elapsed = round(time.time() - started, 2)
        report = result.report
        (case_dir / "report.json").write_text(report.to_json())

        render_views(
            result.volume.array,
            split=result.split,
            tumor_mask=result.prediction.tumor_mask,
            gt_tumor_mask=reference["tumor"] if reference else None,
            out_dir=case_dir,
            case_id=case_id,
        )
        print(f"       [npz] {report.assessment} ({report.confidence}) in {elapsed}s", flush=True)

        entry = {
            "case_id": case_id,
            "shows": intent,
            "cohort": "PanTS official test set (held out)",
            "modality": "contrast-enhanced abdominal CT",
            "inputs": {
                "preprocessed_npz": str(npz_path),
                "raw_nifti": str(raw_path) if raw_path.exists() else None,
            },
            "from_preprocessed_npz": summarize(report, elapsed),
        }

        if reference is not None:
            ref_tumor = reference["tumor"]
            entry["reference"] = {
                "tumor_present": bool(ref_tumor.any()),
                "tumor_location": tumor_location(reference["pancreas"], ref_tumor, volume.spacing),
                "tumor_dice": round(dice(result.prediction.tumor_mask, ref_tumor), 4),
                "pancreas_dice": round(dice(result.prediction.pancreas_mask, reference["pancreas"]), 4),
            }
            print(f"       reference: {entry['reference']}", flush=True)

        # --- raw NIfTI path, exercising preprocess.py --------------------
        if not args.skip_raw and raw_path.exists():
            try:
                started = time.time()
                raw_result = analyzer.analyze(raw_path, case_id=case_id)
                raw_elapsed = round(time.time() - started, 2)
                raw_report = raw_result.report
                (case_dir / "report_from_raw_nifti.json").write_text(raw_report.to_json())
                render_views(
                    raw_result.volume.array,
                    split=raw_result.split,
                    tumor_mask=raw_result.prediction.tumor_mask,
                    out_dir=case_dir / "from_raw_nifti",
                    case_id=case_id,
                )
                entry["from_raw_nifti"] = summarize(raw_report, raw_elapsed)
                agree = raw_report.tumor_detected == report.tumor_detected and (
                    raw_report.dominant_region == report.dominant_region
                )
                entry["paths_agree"] = bool(agree)
                print(f"       [raw] {raw_report.assessment} ({raw_report.confidence}) "
                      f"in {raw_elapsed}s | agrees with npz: {agree}", flush=True)
            except Exception as exc:
                entry["from_raw_nifti_error"] = f"{type(exc).__name__}: {exc}"
                print(f"       [raw] FAILED: {type(exc).__name__}: {exc}", flush=True)

        index.append(entry)

    index_path = out_root / "index.json"
    if wanted and index_path.exists():
        # Partial run: keep the entries for cases that were not re-run.
        previous = json.loads(index_path.read_text())
        done = {e["case_id"] for e in index}
        order = [c for c, _ in CASES]
        # Entries no longer listed in CASES are dropped, so swapping a case out
        # does not leave a stale one behind.
        carried = [e for e in previous if e["case_id"] not in done and e["case_id"] in order]
        index = sorted(index + carried, key=lambda e: order.index(e["case_id"]))
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\n[done] {len(index)} cases -> {out_root}/", flush=True)


if __name__ == "__main__":
    main()
