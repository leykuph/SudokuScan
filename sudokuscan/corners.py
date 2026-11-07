"""Stage 2 - Harris corner detection and automatic selection of the four outer puzzle corners."""

import cv2
import numpy as np
from .preprocess import to_original

def detect_corners(prep):
    """Return (response, candidates, corners):

    response   - Harris response map, same size as prep["work"]
    candidates - (N, 2) array of (x, y) candidate points in working-image pixels
    corners    - (4, 2) array TL, TR, BR, BL in ORIGINAL-image pixels (see preprocess.to_original),
                 or None if no puzzle is found
    """
    blockSize = 7 # A larger window gives a smoother response but merges nearby features
    ksize = 5 # Sobel aperture
    k = 0.05 # is usually 0.04–0.06
    src = prep["work"].astype(np.float32) # Must be float32
    candidates = np.zeros((0, 2))
    t = 0.04
    n = 10


    dst = cv2.cornerHarris(src, blockSize, ksize, k)

    strong = dst > t * dst.max()
    kernel = np.ones((n, n), np.uint8)
    peaks = dst == cv2.dilate(dst, kernel)
    candidates = np.argwhere(strong & peaks)[:, ::-1]

    return dst, candidates, None
