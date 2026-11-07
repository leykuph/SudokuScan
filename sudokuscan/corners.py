"""Stage 2 - Harris corner detection and automatic selection of the four outer puzzle corners."""

import cv2
import numpy as np
from .preprocess import to_original

BLOCK_SIZE = 5  # A larger window gives a smoother response but merges nearby features
KSIZE = 3       # Sobel aperture
K = 0.02        # is usually 0.04–0.06
T = 0.02        # threshold, relative to the strongest response
N = 8           # non-maximum suppression window
ASPECT_MAX = 2    # grid 1.00–1.09; pen 3.28, page outlines 1.5+
FILL_MIN = 0.08     # page outline 0.035, sheet outline 0.051
FILL_MAX = 0.20     # grid ~0.15; crossword 0.245, single letters ~0.5
MIN_SIDE = 0.25     # box side as a fraction of the image's shorter side
MASK_GROW = 7       # dilation kernel; title text is 10–20 px away

def harris_candidates(work, blockSize, ksize, k, t, n):
    """Return (dst, candidates): the Harris response and the (N, 2) candidate points (x, y)."""
    src = work.astype(np.float32)  # Must be float32
    dst = cv2.cornerHarris(src, blockSize, ksize, k)

    strong = dst > t * dst.max()
    kernel = np.ones((n, n), np.uint8)
    peaks = dst == cv2.dilate(dst, kernel)
    candidates = np.argwhere(strong & peaks)[:, ::-1]

    return dst, candidates

def find_grid(ink):
    """Return the label image and the number of the piece that looks like the grid, or None."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    H, W = ink.shape[:2]

    best, best_box = None, 0
    for i in range(1, count):                 # skip background
        x, y, w, h, area = stats[i]
        box = w * h
        aspect = max(w, h) / min(w, h)
        fill = area / box
        touches_edge = x == 0 or y == 0 or x + w == W or y + h == H

        if (min(w, h) >= MIN_SIDE * min(H, W)
                and aspect <= ASPECT_MAX
                and FILL_MIN <= fill <= FILL_MAX
                and not touches_edge
                and box > best_box):
            best, best_box = i, box

    return labels, best


def detect_corners(prep):
    dst, candidates = harris_candidates(prep["work"], BLOCK_SIZE, KSIZE, K, T, N)

    labels, best = find_grid(prep["ink"])
    if best is None:
        return dst, candidates, None

        # Step C: mask of the grid, grown a little
    mask = (labels == best).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((MASK_GROW, MASK_GROW), np.uint8))

        # Step D: keep only candidates on the mask
    on_mask = mask[candidates[:, 1], candidates[:, 0]] > 0
    pts = candidates[on_mask]
    if len(pts) < 4:
        return dst, candidates, None

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    corners = np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                        pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)

    return dst, candidates, to_original(corners, prep["scale"])
