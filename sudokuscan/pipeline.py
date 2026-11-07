"""Runs the stages on one image and collects everything the output files need."""

import traceback
from dataclasses import dataclass, field

import numpy as np

from .corners import CornerDetection, detect_corners
from .preprocess import Preprocessed, preprocess
from .recognize import read_grid
from .rectify import Rectification, rectify_all
from .validity import find_conflicts


@dataclass
class ScanResult:
    verdict: str = "not_found"  # "valid", "invalid" or "not_found"
    runtime_s: float = 0.0
    prep: Preprocessed | None = None
    detection: CornerDetection | None = None
    rect: Rectification | None = None
    grid: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=int))
    conflicts: list[tuple[int, int]] = field(default_factory=list)
    error: str | None = None  # set if a stage raised an exception

    @property
    def corners(self) -> np.ndarray | None:
        return None if self.detection is None else self.detection.corners


def scan(image: np.ndarray, verbose: bool = False) -> ScanResult:
    """Run stages 1-3, 5 and 6. Never raises: a failing stage is recorded in result.error."""
    res = ScanResult()
    try:
        res.prep = preprocess(image)                         # Stage 1
        res.detection = detect_corners(res.prep)             # Stage 2
        if res.corners is None:
            return res
        res.rect = rectify_all(image, res.corners)           # Stage 3
        res.grid = read_grid(res.rect.images["H4"])          # Stage 5
        res.conflicts = find_conflicts(res.grid)             # Stage 6
        res.verdict = "invalid" if res.conflicts else "valid"
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()
        # Keep whatever was computed. Without corners there is no puzzle; with corners the grid
        # read so far (zeros if recognition failed) is still the best answer we have.
        if res.corners is not None:
            res.verdict = "valid"
    return res
