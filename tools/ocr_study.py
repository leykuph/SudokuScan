"""OCR study for Stage 5: accuracy on the filled cells for several Tesseract settings, with and without
cell preparation. Run sudoku_scan.py --batch first; this reads the <name>_rect_H4.png files.

    python tools/ocr_study.py results/ SudokuScan_devset/annotations
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan import recognize as R  # noqa: E402

WHITELIST = "-c tessedit_char_whitelist=123456789"
SETTINGS = [  # (label, prepare the cell?, tesseract config)
    ("raw crop, psm 10", False, f"--psm 10 {WHITELIST}"),
    ("prepared, psm 10", True, f"--psm 10 {WHITELIST}"),
    ("prepared, psm 8", True, f"--psm 8 {WHITELIST}"),
    ("prepared, psm 7", True, f"--psm 7 {WHITELIST}"),
    ("raw crop, psm 13", False, f"--psm 13 {WHITELIST}"),
    ("prepared, psm 13", True, f"--psm 13 {WHITELIST}"),
    ("prepared, psm 10, no whitelist", True, "--psm 10"),
]


def filled_cells(results, annotations):
    """(true digit, prepared image, raw grey crop) for every filled cell the empty test also calls filled."""
    cells = []
    for ann_path in sorted(Path(annotations).glob("*.json")):
        ann = json.loads(ann_path.read_text())
        rect = cv2.imread(str(Path(results) / f"{Path(ann['image']).stem}_rect_H4.png"))
        if rect is None:
            continue
        gray, ink = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), R.binarize(rect)
        for row in range(9):
            for col in range(9):
                digit = R.find_digit(R.crop(ink, row, col))
                if ann["grid"][row][col] and digit is not None:
                    cells.append((ann["grid"][row][col], R.prepare(digit), R.crop(gray, row, col)))
    return cells


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cells = filled_cells(sys.argv[1], sys.argv[2])
    print(f"{len(cells)} filled cells\n")
    print(f"{'setting':<34}{'accuracy':>9}{'no answer':>11}{'wrong':>7}{'ms/cell':>9}")
    for label, prepared, config in SETTINGS:
        R.TESSERACT = config
        start = time.perf_counter()
        preds = [R.ocr(prep if prepared else raw) for _, prep, raw in cells]
        ms = 1000 * (time.perf_counter() - start) / len(cells)
        truth = [t for t, _, _ in cells]
        correct = sum(p == t for p, t in zip(preds, truth))
        blank = sum(p == 0 for p in preds)
        print(f"{label:<34}{correct / len(cells):>9.3f}{blank:>11}{len(cells) - correct - blank:>7}{ms:>9.0f}")
        confusions = Counter((t, p) for p, t in zip(preds, truth) if p != t and p != 0)
        if confusions:
            print(f"{'':<34}confusions: " + ", ".join(f"{t}->{p} x{n}" for (t, p), n in confusions.most_common(4)))


if __name__ == "__main__":
    main()
