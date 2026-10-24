"""End-to-end CLI: a scan in, a report and overlay images out.

    python scripts/run_inference.py --input case.nii.gz --out_dir out/

Use it to smoke-test the package and as a worked example of the call sequence
a web backend should make.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panguide3d import PanGuide3D            # noqa: E402
from panguide3d.visualize import render_views      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="NIfTI file, DICOM series directory, or a preprocessed .npz")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--weights", default=None, help="defaults to checkpoints/")
    ap.add_argument("--device", default=None, help="cuda or cpu; auto-detected by default")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--no_images", action="store_true", help="write the report only")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = PanGuide3D(weights_path=args.weights, device=args.device)
    print(f"[model] {analyzer.info.get('model_name', 'PanGuide3D')} "
          f"on {analyzer.device}")

    source = Path(args.input)
    if source.suffix == ".npz":
        # Already preprocessed by the research repo - skip resampling.
        import numpy as np
        from panguide3d.preprocess import Volume
        npz = np.load(source, allow_pickle=True)
        volume = Volume(
            array=npz["image"].astype("float32"),
            spacing=tuple(float(s) for s in npz["spacing"]),
            origin=(0.0, 0.0, 0.0), direction=tuple(),
            original_spacing=tuple(float(s) for s in npz["spacing"]),
            original_size=tuple(int(v) for v in npz["image"].shape[::-1]),
            case_id=source.stem,
        )
        result = analyzer._run(volume, threshold=args.threshold, overlap=args.overlap)
    else:
        result = analyzer.analyze(source, threshold=args.threshold, overlap=args.overlap)

    report = result.report
    (out_dir / f"{report.case_id}_report.json").write_text(report.to_json())

    print(f"\n=== {report.case_id} ===")
    print(f"{report.assessment}  (confidence: {report.confidence})")
    print(f"pancreas {report.pancreas_volume_cm3} cm3   tumor {report.tumor_volume_cm3} cm3")
    print(f"regions: {json.dumps(report.region_volumes_cm3)}")
    print("\nreasons:")
    for r in report.reasons:
        print(f"  - {r}")
    print("\ncautions:")
    for c in report.cautions:
        print(f"  - {c}")
    print(f"\n[{result.elapsed_seconds}s]  report -> {out_dir / (report.case_id + '_report.json')}")

    if not args.no_images:
        written = render_views(
            result.volume.array, split=result.split,
            tumor_mask=result.prediction.tumor_mask,
            out_dir=out_dir, case_id=report.case_id,
        )
        for view, path in written.items():
            print(f"  {view:<9} -> {path}")


if __name__ == "__main__":
    main()
