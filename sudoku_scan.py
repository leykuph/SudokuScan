"""SudokuScan: photo of a printed sudoku -> 9x9 grid.

    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sudokuscan.corners import detect_corners
from sudokuscan.preprocess import preprocess, to_work
from sudokuscan.recognize import read_grid
from sudokuscan.rectify import SIZE, rectify
from sudokuscan.validity import find_conflicts

CELL = SIZE // 9


def scan(image):
    """Run all stages. Never crashes: if a stage fails, the result keeps what was computed so far."""
    r: dict[str, Any] = {"verdict": "not_found", "corners": None, "grid": np.zeros((9, 9), int), "conflicts": [],
         "prep": None, "harris": None, "rect": None, "error": None}
    try:
        if image is None:
            raise ValueError("could not read the image")
        r["prep"] = preprocess(image)                       # Stage 1
        r["harris"] = detect_corners(r["prep"])             # Stage 2: (response, candidates, corners)
        r["corners"] = r["harris"][2]
        if r["corners"] is None:
            return r
        r["rect"] = rectify(image, r["corners"])            # Stage 3
        r["grid"] = read_grid(r["rect"]["images"]["H4"])    # Stage 5
        r["conflicts"] = find_conflicts(r["grid"])          # Stage 6
        r["verdict"] = "invalid" if r["conflicts"] else "valid"
    except Exception as e:
        traceback.print_exc()
        r["error"] = f"{type(e).__name__}: {e}"
        if r["corners"] is not None:   # puzzle found, a later stage failed: report the grid we have
            r["verdict"] = "valid"
    return r


def draw_harris(prep, response, candidates, corners):
    """Harris response map with all candidates (green) and the four selected corners (red)."""
    # The response spans several orders of magnitude; the 0.25 power makes weak corners visible.
    r = np.maximum(response, 0) / max(response.max(), 1e-12)
    img = cv2.cvtColor((255 * r ** 0.25).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for x, y in candidates:
        cv2.circle(img, (round(x), round(y)), 4, (0, 255, 0), 1)
    if corners is not None:
        pts = to_work(corners, prep["scale"])
        cv2.polylines(img, [pts.round().astype(np.int32)], True, (0, 0, 255), 2)
        for label, (x, y) in zip(["TL", "TR", "BR", "BL"], pts):
            cv2.putText(img, label, (round(x) + 8, round(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def draw_debug(rectified, grid, conflicts):
    """Rectified H4 image with cell boundaries, recognised digits (blue) and conflicting cells (red)."""
    img = rectified.copy()
    for i in range(10):
        p = min(i * CELL, SIZE - 1)
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (p, 0), (p, SIZE - 1), (0, 160, 0), thickness)
        cv2.line(img, (0, p), (SIZE - 1, p), (0, 160, 0), thickness)
    for row in range(9):
        for col in range(9):
            if grid[row][col]:   # small digit in the top-left, so the printed digit stays visible
                cv2.putText(img, str(grid[row][col]), (col * CELL + 3, row * CELL + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for row, col in conflicts:
        cv2.rectangle(img, (col * CELL + 2, row * CELL + 2), ((col + 1) * CELL - 3, (row + 1) * CELL - 3), (0, 0, 255), 3)
    return img


def as_list(m):
    return None if m is None else np.round(np.asarray(m, dtype=float), 4).tolist()


def save(outdir, path, r, runtime):
    name = path.stem
    rect = r["rect"] or {}
    grid = np.asarray(r["grid"], dtype=int)

    (outdir / f"{name}_grid.txt").write_text("".join("".join(map(str, row)) + "\n" for row in grid))

    result = {
        "image": path.name,
        "corners": as_list(r["corners"]),
        "A3": as_list(rect.get("A3")),
        "A4": as_list(rect.get("A4")),
        "H4": as_list(rect.get("H4")),
        "a3_corner_err_px": rect.get("a3_corner_err_px"),
        "a4_rms_px": rect.get("a4_rms_px"),
        "grid": grid.tolist(),
        "verdict": r["verdict"],
        "conflicts": [[int(row), int(col)] for row, col in r["conflicts"]],
        "runtime_s": round(runtime, 3),
        "error": r["error"],
    }
    # One key per line, so the file stays readable.
    lines = [f'  "{key}": {json.dumps(value)}' for key, value in result.items()]
    (outdir / f"{name}_result.json").write_text("{\n" + ",\n".join(lines) + "\n}\n")

    if r["harris"] is not None:
        cv2.imwrite(str(outdir / f"{name}_harris.png"), draw_harris(r["prep"], *r["harris"]))
    if rect:
        for model in ("A3", "A4", "H4"):
            cv2.imwrite(str(outdir / f"{name}_rect_{model}.png"), rect["images"][model])
        cv2.imwrite(str(outdir / f"{name}_debug.png"), draw_debug(rect["images"]["H4"], grid, r["conflicts"]))


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="one photo")
    source.add_argument("--batch", type=Path, help="a folder of photos")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    if args.input:
        paths = [args.input]
    else:
        paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        start = time.perf_counter()
        image = cv2.imread(str(path))   # imread applies the EXIF rotation of phone photos
        r = scan(image)
        runtime = time.perf_counter() - start
        save(args.outdir, path, r, runtime)
        print(f"{path.name}: {r['verdict']} ({runtime:.2f} s)")


if __name__ == "__main__":
    main()
