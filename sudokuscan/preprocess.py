"""Stage 1 - preprocessing: greyscale, noise reduction, illumination normalisation, downscaling."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Preprocessed:
    image: np.ndarray  # original BGR image, full resolution
    work: np.ndarray   # normalised greyscale image used for corner detection
    scale: float       # work-image pixels per original-image pixel (work = original * scale)


def preprocess(image: np.ndarray) -> Preprocessed:
    raise NotImplementedError("Stage 1 (preprocessing) is not implemented yet")
