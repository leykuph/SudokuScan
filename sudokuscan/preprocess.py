"""Stage 1 - preprocessing: downscaling, greyscale, noise reduction, illumination normalisation.

Every photo is resized to a fixed working size (LONG_SIDE px on its long side), so all kernel sizes
below are fixed fractions of the image size and the same code behaves the same on 1000 px and
4000 px inputs.
"""

from dataclasses import dataclass

import cv2
import numpy as np

LONG_SIDE = 1000             # working-image long side in pixels

# All sizes as fractions of the working-image long side.
DENOISE_SIGMA = 0.001        # Gaussian noise reduction, ~1 px
BACKGROUND_KERNEL = 0.015    # closing kernel, wider than any pen stroke or grid line (~15 px)
BACKGROUND_SIGMA = 0.015     # smoothing of the background estimate
INK_BLOCK = 0.025            # adaptive-threshold neighbourhood (~25 px, about one cell or less)

MAX_RELATIVE_GAIN = 3.0      # brightest a region can be lifted relative to well-lit paper
BRIGHT_PAPER_PERCENTILE = 95 # background percentile taken as "well-lit paper"
INK_OFFSET = 10              # a pixel is ink if it is this much darker than its neighbourhood mean


@dataclass
class Preprocessed:
    image: np.ndarray  # original BGR image, full resolution
    work: np.ndarray   # uint8 greyscale, illumination-normalised, LONG_SIDE px long side (Harris input)
    ink: np.ndarray    # uint8 binary, same size as work, ink = 255, paper = 0
    scale: float       # working-image pixels per original-image pixel

    def to_work(self, pts: np.ndarray) -> np.ndarray:
        """Original-image (x, y) points to working-image coordinates."""
        return (np.asarray(pts, dtype=float) + 0.5) * self.scale - 0.5

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        """Working-image (x, y) points to original-image coordinates."""
        return (np.asarray(pts, dtype=float) + 0.5) / self.scale - 0.5


def _odd(n: float) -> int:
    n = max(3, round(n))
    return n if n % 2 else n + 1


def estimate_background(gray: np.ndarray) -> np.ndarray:
    """Paper brightness without the ink: a closing removes dark strokes, a blur smooths the result."""
    k = _odd(BACKGROUND_KERNEL * LONG_SIDE)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.GaussianBlur(closed, (0, 0), BACKGROUND_SIGMA * LONG_SIDE).astype(np.float32)


def normalise_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by the background so shadowed paper becomes as bright as lit paper.

    The gain is capped relative to well-lit paper: shadows need up to ~2.6x, while a dark desk would
    otherwise get 6-9x and its texture would turn into strong false corners.
    """
    background = estimate_background(gray)
    paper = max(float(np.percentile(background, BRIGHT_PAPER_PERCENTILE)), 1.0)
    gain = np.minimum(paper / np.maximum(background, 1.0), MAX_RELATIVE_GAIN)
    return np.clip(gray * gain * (255.0 / paper), 0, 255).astype(np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink map: pixels darker than their local neighbourhood by INK_OFFSET."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 _odd(INK_BLOCK * LONG_SIDE), INK_OFFSET)


def preprocess(image: np.ndarray) -> Preprocessed:
    scale = LONG_SIDE / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), DENOISE_SIGMA * LONG_SIDE)

    return Preprocessed(image=image, work=normalise_illumination(gray), ink=ink_mask(gray), scale=scale)
