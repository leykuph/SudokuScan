"""Stage 2 - Harris corner detection and automatic selection of the four outer puzzle corners."""


def detect_corners(prep):
    """Return (response, candidates, corners):

    response   - Harris response map, same size as prep["work"]
    candidates - (N, 2) array of (x, y) candidate points in working-image pixels
    corners    - (4, 2) array TL, TR, BR, BL in ORIGINAL-image pixels (see preprocess.to_original),
                 or None if no puzzle is found
    """
    raise NotImplementedError("Stage 2 (corner detection) is not implemented yet")
