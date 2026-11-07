"""Stage 5 - cell extraction, empty-cell detection and digit recognition."""

import cv2
import numpy as np
import pytesseract

CELL = 50          # px per cell in the 450x450 rectified image
MARGIN = 7         # px cut from each side of a cell, so grid lines don't reach the OCR (digits are 20-28 px tall)
MIN_HEIGHT = 12    # a blob shorter than this is noise, not a digit (digits are >= 20 px)
EDGE_BAND = 6      # blobs whose centre is this close to the crop edge are grid-line leftovers
OCR_HEIGHT = 48    # digits are scaled to this height before OCR
# psm 13 = "raw line": no layout analysis at all. On the dev set psm 13 reads 99.3% of prepared cells;
# psm 10 (single character) returns nothing for 28 of 565 clear digits (see tools/ocr_study.py).
TESSERACT = "--psm 13 -c tessedit_char_whitelist=123456789"


def binarize(rectified):
    """Ink = 255. Adaptive threshold over about one cell, so shadows across the grid don't matter."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 15)


def crop(img, row, col):
    """The inside of one cell, without MARGIN px on each side."""
    y, x = row * CELL, col * CELL
    return img[y + MARGIN:y + CELL - MARGIN, x + MARGIN:x + CELL - MARGIN]


def find_digit(ink):
    """Return the digit as a binary image (ink = 255), or None if the cell is empty.

    Empty-cell criterion: there is no ink blob that is at least MIN_HEIGHT px tall and has its
    centre away from the crop edge. Leftovers of grid lines are thin strips along the edge."""
    count, labels, stats, centres = cv2.connectedComponentsWithStats(ink)
    h, w = ink.shape
    best = None
    for i in range(1, count):
        cx, cy = centres[i]
        near_edge = min(cx, cy, w - cx, h - cy) < EDGE_BAND
        if stats[i, 3] >= MIN_HEIGHT and not near_edge:
            if best is None or stats[i, 4] > stats[best, 4]:
                best = i
    if best is None:
        return None
    x, y, bw, bh, _ = stats[best]
    return (labels[y:y + bh, x:x + bw] == best).astype(np.uint8) * 255


def prepare(digit):
    """OCR input: the digit alone, scaled to OCR_HEIGHT, black on white, centred with white padding."""
    h, w = digit.shape
    scale = OCR_HEIGHT / h
    digit = cv2.resize(digit, (max(1, round(w * scale)), OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
    pad = OCR_HEIGHT // 2
    return cv2.copyMakeBorder(255 - digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def ocr(img):
    """Tesseract on one prepared cell; 0 if it returns no digit."""
    text = str(pytesseract.image_to_string(img, config=TESSERACT)).strip()
    return int(text[0]) if text[:1] in list("123456789") else 0


def read_grid(rectified, prepare_cells=True):
    """Return a 9x9 int array of the givens in a 450x450 rectified image, 0 for empty cells.

    prepare_cells=False sends the raw grey cell crop to the OCR instead, for the with/without comparison."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    ink = binarize(rectified)
    grid = np.zeros((9, 9), int)
    for row in range(9):
        for col in range(9):
            digit = find_digit(crop(ink, row, col))   # empty test first, before any OCR
            if digit is None:
                continue
            grid[row, col] = ocr(prepare(digit) if prepare_cells else crop(gray, row, col))
    return grid
