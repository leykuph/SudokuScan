"""Stage 3 - rectification with A3 (affine, 3 points), A4 (affine, 4 points) and H4 (homography)."""

from typing import Any

import cv2
import numpy as np

SIZE = 450  # 9 cells x 50 px

# Target corners TL, TR, BR, BL. Pixel centres sit at whole numbers, so the outer edge of a
# 450 px image runs from -0.5 to 449.5; mapping the corners there makes every cell exactly 50x50.
TARGET = np.array([[-0.5, -0.5], [SIZE - 0.5, -0.5], [SIZE - 0.5, SIZE - 0.5], [-0.5, SIZE - 0.5]], dtype=np.float32)
TL, TR, BR, BL = 0, 1, 2, 3


def apply_affine(A, pts):
    """Map (N, 2) points with a 2x3 affine matrix: [x', y'] = A @ [x, y, 1]."""
    return pts @ A[:, :2].T + A[:, 2]


def rectify(image, corners) -> dict[str, Any]:
    """Return {"A3", "A4": 2x3 matrices, "H4": 3x3 matrix, "images": {"A3", "A4", "H4": 450x450 images},
    "a3_corner_err_px", "a4_rms_px", "a3_br_px", "h4_max_err_px"}."""
    src = np.asarray(corners, dtype=np.float32)

    # A3: 6 unknowns, 3 point pairs x 2 equations = exactly determined. BR is not used.
    A3 = cv2.getAffineTransform(src[[TL, TR, BL]], TARGET[[TL, TR, BL]])

    # A4: 6 unknowns, 4 point pairs = 8 equations, solved by least squares. estimateAffine2D uses
    # RANSAC by default, which drops one of the 4 corners as an "outlier"; a huge threshold keeps
    # all four as inliers, and the final refinement is then a plain least-squares fit.
    A4, _ = cv2.estimateAffine2D(src, TARGET, method=cv2.RANSAC, ransacReprojThreshold=1e6)

    # H4: 8 unknowns (3x3 up to scale), 4 point pairs = 8 equations = exactly determined.
    H4 = cv2.getPerspectiveTransform(src, TARGET)

    images = {
        "A3": cv2.warpAffine(image, A3, (SIZE, SIZE)),
        "A4": cv2.warpAffine(image, A4, (SIZE, SIZE)),
        "H4": cv2.warpPerspective(image, H4, (SIZE, SIZE)),
    }

    # Where A3 puts the unused BR corner, and how far that is from where it should be.
    a3_br = apply_affine(A3, src[BR])
    a3_err = np.linalg.norm(a3_br - TARGET[BR])

    # A4 residual: RMS distance between the mapped corners and the target corners.
    a4_res = np.linalg.norm(apply_affine(A4, src) - TARGET, axis=1)
    a4_rms = np.sqrt(np.mean(a4_res ** 2))

    # H4 should hit all four target corners exactly (up to rounding).
    h4_mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H4).reshape(-1, 2)
    h4_err = np.linalg.norm(h4_mapped - TARGET, axis=1).max()

    return {"A3": A3, "A4": A4, "H4": H4, "images": images,
            "a3_corner_err_px": float(a3_err), "a4_rms_px": float(a4_rms),
            "a3_br_px": a3_br.tolist(), "h4_max_err_px": float(h4_err)}
