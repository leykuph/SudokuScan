"""Stage 3 - rectification with the A3 (affine, 3 points), A4 (affine, 4 points) and H4 (homography) models."""

from dataclasses import dataclass

import numpy as np

TARGET_SIZE = 450  # 9 cells x 50 px


@dataclass
class Rectification:
    A3: np.ndarray                 # 2x3 affine from TL, TR, BL
    A4: np.ndarray                 # 2x3 least-squares affine from all four corners
    H4: np.ndarray                 # 3x3 homography from all four corners
    images: dict[str, np.ndarray]  # "A3", "A4", "H4" -> 450x450 BGR rectified image
    a3_corner_err_px: float        # distance between where A3 maps BR and the target BR
    a4_rms_px: float               # RMS residual of A4 over the four corners


def rectify_all(image: np.ndarray, corners: np.ndarray) -> Rectification:
    raise NotImplementedError("Stage 3 (rectification) is not implemented yet")
