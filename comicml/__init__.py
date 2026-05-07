"""
comicml — CNN page-corner detection package.

Public API preserved from the original flat `comicml.py` module so that
existing imports (`from comicml import detect_page_bounds_hybrid`, etc.)
continue to work unchanged after the refactor.
"""

from .inference import (
    # Primary entry point used by comicscans.py, comiceval.py, webapp/scan.
    detect_page_bounds_hybrid,
    # Corner prediction primitives.
    predict_corners,
    predict_corners_with_disagreement,
    predict_corners_ensemble,
    # Refinement (line-fit & profile-based edge snapping).
    refine_corners,
    refine_corners_linefit,
    # Ensemble plumbing.
    _resolve_ensemble_paths,
    _get_cached_model,
    _get_cached_ensemble,
    # Constants.
    MODEL_FILE,
    ENSEMBLE_MODELS,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "detect_page_bounds_hybrid",
    "predict_corners",
    "predict_corners_with_disagreement",
    "predict_corners_ensemble",
    "refine_corners",
    "refine_corners_linefit",
    "_resolve_ensemble_paths",
    "_get_cached_model",
    "_get_cached_ensemble",
    "MODEL_FILE",
    "ENSEMBLE_MODELS",
    "INPUT_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
