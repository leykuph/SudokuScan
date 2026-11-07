"""Stage 2 - Harris corner detection and automatic selection of the four outer puzzle corners."""

from dataclasses import dataclass

import numpy as np

from .preprocess import Preprocessed


@dataclass
class CornerDetection:
    response: np.ndarray        # Harris response map, same size as prep.work
    candidates: np.ndarray      # (N, 2) candidate points (x, y) in work-image pixels
    corners: np.ndarray | None  # (4, 2) TL, TR, BR, BL in ORIGINAL-image pixels, None if not found


def detect_corners(prep: Preprocessed) -> CornerDetection:
    raise NotImplementedError("Stage 2 (corner detection) is not implemented yet")
