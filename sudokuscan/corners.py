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
    """Return (response, candidates, corners):

    response   - Harris response map, same size as prep["work"]
    candidates - (N, 2) array of (x, y) candidate points in working-image pixels
    corners    - (4, 2) array TL, TR, BR, BL in ORIGINAL-image pixels (see preprocess.to_original),
                 or None if no puzzle is found
    """
    dst, candidates = harris_candidates(prep["work"], BLOCK_SIZE, KSIZE, K, T, N)

    return dst, candidates, None
