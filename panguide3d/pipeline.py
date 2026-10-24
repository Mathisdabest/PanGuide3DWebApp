"""One call from an uploaded file to a report plus overlay-ready masks.

This is the entry point a web backend should use::

    from panguide3d import PanGuide3D

    analyzer = PanGuide3D()          # loads weights once, at startup
    result = analyzer.analyze("/tmp/upload.nii.gz")
    result.report.to_json()
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .model import load_model, model_info
from .predict import Prediction, predict
from .preprocess import Volume, preprocess, preprocess_array
from .regions import RegionSplit, split_pancreas
from .report import Report, build_report


@dataclass
class AnalysisResult:
    """Everything one scan produced, ready for an API response or a viewer."""

    volume: Volume
    prediction: Prediction
    split: RegionSplit
    report: Report
    elapsed_seconds: float


class PanGuide3D:
    """Holds the loaded model so the weights are read once per process."""

    def __init__(self, weights_path=None, device: str | None = None):
        self.model, self.device = load_model(weights_path, device)
        self.info = model_info(weights_path)

    def analyze(self, path, case_id: str | None = None, **predict_kwargs) -> AnalysisResult:
        """Preprocess, segment and report on a scan file or DICOM directory."""
        return self._run(preprocess(path, case_id=case_id), **predict_kwargs)

    def analyze_array(self, array_zyx, spacing_xyz, case_id: str = "case", **predict_kwargs):
        """Same, for a volume already in memory in raw Hounsfield units."""
        return self._run(preprocess_array(array_zyx, spacing_xyz, case_id), **predict_kwargs)

    def _run(self, volume: Volume, **predict_kwargs) -> AnalysisResult:
        started = time.time()
        prediction = predict(self.model, volume, device=self.device, **predict_kwargs)
        split = split_pancreas(prediction.pancreas_mask, prediction.spacing)
        report = build_report(volume, prediction, split)
        return AnalysisResult(
            volume=volume,
            prediction=prediction,
            split=split,
            report=report,
            elapsed_seconds=round(time.time() - started, 2),
        )
