"""Stage 1 - preprocessing."""

from dataclasses import dataclass

import cv2
import numpy as np

LONG_SIDE = 1000  # every photo is resized to this long side; all sizes below are fractions of it
MAX_GAIN = 3      # never brighten a region more than 3x relative to well-lit paper


@dataclass
class Preprocessed:
    image: np.ndarray  # original photo
    work: np.ndarray   # normalised greyscale, Harris input
    ink: np.ndarray    # binary, ink = 255, paper = 0
    scale: float       # work pixels per original pixel

    def to_work(self, pts):
        return (np.asarray(pts, dtype=float) + 0.5) * self.scale - 0.5

    def to_original(self, pts):
        return (np.asarray(pts, dtype=float) + 0.5) / self.scale - 0.5


def odd(n):
    n = max(3, round(n))
    return n if n % 2 == 1 else n + 1


def preprocess(image):
    # 1. Resize to a fixed size, so the same parameters work on 1000 px and 4000 px photos.
    scale = LONG_SIDE / max(image.shape[:2])
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    # 2. Greyscale and a light blur against sensor noise.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.001 * LONG_SIDE)

    # 3. Background = paper brightness without ink. The closing erases dark strokes thinner than
    #    the kernel (grid lines, digits); the blur smooths what is left.
    k = odd(0.015 * LONG_SIDE)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), 0.015 * LONG_SIDE)

    # 4. Divide by the background so shadowed paper becomes as bright as lit paper. Clamping the
    #    background at paper / MAX_GAIN limits the gain, so a dark desk is not blown up into texture.
    paper = np.percentile(bg, 95)
    bg = np.maximum(bg, paper / MAX_GAIN).astype(np.uint8)
    work = cv2.divide(gray, bg, scale=255)

    # 5. Ink: pixels at least 10 grey levels darker than the mean of their neighbourhood.
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                odd(0.025 * LONG_SIDE), 10)

    return Preprocessed(image, work, ink, scale)
