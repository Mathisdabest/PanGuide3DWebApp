"""Pancreas CT segmentation and reporting for the diagnosis web app.

Research code for decision support. Not a medical device, and not cleared for
clinical use - see the package README.
"""
from .pipeline import AnalysisResult, PanGuide3D
from .predict import Prediction, predict
from .preprocess import Volume, preprocess, preprocess_array
from .regions import RegionSplit, region_overlap, region_volumes, split_pancreas, tumor_location
from .report import Report, build_report
from .model import build_model, load_model, model_info

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult", "PanGuide3D",
    "Volume", "preprocess", "preprocess_array",
    "Prediction", "predict",
    "RegionSplit", "split_pancreas", "tumor_location", "region_overlap", "region_volumes",
    "Report", "build_report",
    "build_model", "load_model", "model_info",
]
