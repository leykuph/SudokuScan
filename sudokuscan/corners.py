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
MIN_SIDE = 0.25     # box side as a fraction of the image's shorter side
MASK_GROW = 7       # dilation kernel; title text is 10–20 px away
MIN_HOLES = 4       # a frame has 0 holes, a lone box 1

def count_holes(labels, i, x, y, w, h):
    """Number of enclosed holes in component i, using only its bounding box."""
    piece = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
    _, hierarchy = cv2.findContours(piece, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(np.sum(hierarchy[0][:, 3] != -1))   # has a parent -> hole



def harris_candidates(work, blockSize, ksize, k, t, n, mask=None):
    src = work.astype(np.float32)  # Must be float32
    dst = cv2.cornerHarris(src, blockSize, ksize, k)

    ref = dst[mask > 0].max() if mask is not None else dst.max()
    strong = dst > t * ref
    kernel = np.ones((n, n), np.uint8)
    peaks = dst == cv2.dilate(dst, kernel)
    candidates = np.argwhere(strong & peaks)[:, ::-1]

    return dst, candidates


def find_grid(ink):
    """Return the label image and the number of the piece that looks like the grid, or None."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    H, W = ink.shape[:2]

    best, best_holes = None, 0
    for i in range(1, count):                 # skip background
        x, y, w, h, area = stats[i]
        aspect = max(w, h) / min(w, h)
        touches_edge = x == 0 or y == 0 or x + w == W or y + h == H

        if (min(w, h) < MIN_SIDE * min(H, W)
                or aspect > ASPECT_MAX
                or touches_edge):
            continue

        holes = count_holes(labels, i, x, y, w, h)
        if holes >= MIN_HOLES and holes > best_holes:
            best, best_holes = i, holes

    return labels, best


def detect_corners(prep):
    labels, best = find_grid(prep["ink"])
    if best is None:
        dst, candidates = harris_candidates(prep["work"], BLOCK_SIZE, KSIZE, K, T, N)
        return dst, candidates, None

    mask = (labels == best).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((MASK_GROW, MASK_GROW), np.uint8))

    dst, candidates = harris_candidates(prep["work"], BLOCK_SIZE, KSIZE, K, T, N, mask=mask)

    on_mask = mask[candidates[:, 1], candidates[:, 0]] > 0
    pts = candidates[on_mask]
    if len(pts) < 4:
        return dst, candidates, None

    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    corners = np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                        pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)

    return dst, candidates, to_original(corners, prep["scale"])
