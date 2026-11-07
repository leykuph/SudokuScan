"""Stage 2 - Harris corner detection and automatic selection of the four outer puzzle corners."""

import cv2
import numpy as np
from .preprocess import to_original

BLOCK_SIZE = 5  # A larger window gives a smoother response but merges nearby features
KSIZE = 3       # Sobel aperture
K = 0.02        # is usually 0.04–0.06
T = 0.02        # threshold, relative to the strongest response
N = 8           # non-maximum suppression window


def harris_candidates(work, blockSize, ksize, k, t, n):
    """Return (dst, candidates): the Harris response and the (N, 2) candidate points (x, y)."""
    src = work.astype(np.float32)  # Must be float32
    dst = cv2.cornerHarris(src, blockSize, ksize, k)

    strong = dst > t * dst.max()
    kernel = np.ones((n, n), np.uint8)
    peaks = dst == cv2.dilate(dst, kernel)
    candidates = np.argwhere(strong & peaks)[:, ::-1]

    return dst, candidates

def detect_corners(prep):
    dst, candidates = harris_candidates(prep["work"], BLOCK_SIZE, KSIZE, K, T, N)

    if len(candidates) < 4:
        return dst, candidates, None

    s = candidates[:, 0] + candidates[:, 1]   # x + y
    d = candidates[:, 0] - candidates[:, 1]   # x - y

    tl = candidates[np.argmin(s)]
    br = candidates[np.argmax(s)]
    bl = candidates[np.argmin(d)]
    tr = candidates[np.argmax(d)]

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    corners = to_original(corners, prep["scale"])

    return dst, candidates, corners
