"""Stage 3 - rectification with A3 (affine, 3 points), A4 (affine, 4 points) and H4 (homography)."""

SIZE = 450  # 9 cells x 50 px


def rectify(image, corners):
    """Return {"A3", "A4": 2x3 matrices, "H4": 3x3 matrix, "images": {"A3", "A4", "H4": 450x450 images},
    "a3_corner_err_px": where A3 puts BR vs the target BR, "a4_rms_px": A4 RMS residual over the corners}."""
    raise NotImplementedError("Stage 3 (rectification) is not implemented yet")
