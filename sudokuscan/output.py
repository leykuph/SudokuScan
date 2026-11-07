"""Writes the per-image output files in the format of Section 6 / Appendix A."""

import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .corners import CornerDetection
from .pipeline import ScanResult
from .preprocess import Preprocessed
from .rectify import TARGET_SIZE, Rectification

CELL = TARGET_SIZE // 9
LABELS = ("TL", "TR", "BR", "BL")


def write_outputs(outdir: Path, image_name: str, res: ScanResult) -> None:
    name = Path(image_name).stem
    (outdir / f"{name}_grid.txt").write_text(format_grid(res.grid))
    (outdir / f"{name}_result.json").write_text(to_json_text(result_to_json(image_name, res)))
    if res.prep is not None and res.detection is not None:
        cv2.imwrite(str(outdir / f"{name}_harris.png"), draw_harris(res.prep, res.detection))
    if res.rect is not None:
        for model, rectified in res.rect.images.items():
            cv2.imwrite(str(outdir / f"{name}_rect_{model}.png"), rectified)
        cv2.imwrite(str(outdir / f"{name}_debug.png"), draw_debug(res.rect, res))


def format_grid(grid: np.ndarray) -> str:
    return "".join("".join(str(int(d)) for d in row) + "\n" for row in grid)


def _matrix(m: Any) -> list[Any] | None:
    return None if m is None else np.asarray(m, dtype=float).tolist()


def _number(x) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def result_to_json(image_name: str, res: ScanResult) -> dict[str, Any]:
    rect = res.rect
    out = {
        "image": image_name,
        "corners": _matrix(res.corners),
        "A3": _matrix(rect.A3) if rect else None,
        "A4": _matrix(rect.A4) if rect else None,
        "H4": _matrix(rect.H4) if rect else None,
        "a3_corner_err_px": _number(rect.a3_corner_err_px) if rect else None,
        "a4_rms_px": _number(rect.a4_rms_px) if rect else None,
        "grid": np.asarray(res.grid, dtype=int).tolist(),
        "verdict": res.verdict,
        "conflicts": [[int(r), int(c)] for r, c in res.conflicts],
        "runtime_s": round(res.runtime_s, 3),
    }
    if res.error:
        out["error"] = res.error
    return out


def to_json_text(data: dict[str, Any]) -> str:
    """Indented JSON, but with each innermost list (a corner, a matrix row, a grid row) on one line."""
    text = json.dumps(data, indent=2)
    return re.sub(r"\[\s+([^\[\]]*?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text) + "\n"


def draw_harris(prep: Preprocessed, det: CornerDetection) -> np.ndarray:
    """Harris response map with all candidates (green) and the four selected corners (red)."""
    r = np.maximum(det.response, 0)
    if r.max() > 0:
        r = r / r.max()
    # The response spans several orders of magnitude; a strong gamma makes weak corners visible.
    vis = cv2.cvtColor((255 * r ** 0.25).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    rad = max(2, round(min(vis.shape[:2]) / 250))
    for x, y in np.asarray(det.candidates).reshape(-1, 2):
        cv2.circle(vis, (round(x), round(y)), rad, (0, 255, 0), 1, cv2.LINE_AA)

    if det.corners is not None:
        pts = prep.to_work(det.corners)
        cv2.polylines(vis, [pts.round().astype(np.int32)], True, (0, 0, 255), max(1, rad // 2), cv2.LINE_AA)
        for label, (x, y) in zip(LABELS, pts):
            p = (round(x), round(y))
            cv2.circle(vis, p, 3 * rad, (0, 0, 255), max(1, rad // 2), cv2.LINE_AA)
            cv2.putText(vis, label, (p[0] + 4 * rad, p[1] - 2 * rad), cv2.FONT_HERSHEY_SIMPLEX,
                        rad / 5, (0, 0, 255), max(1, rad // 2), cv2.LINE_AA)
    return vis


def draw_debug(rect: Rectification, res: ScanResult) -> np.ndarray:
    """Rectified H4 image with cell boundaries, recognised digits (blue), conflicts (red) and verdict."""
    img = rect.images["H4"].copy()

    overlay = img.copy()
    for r, c in res.conflicts:
        cv2.rectangle(overlay, (c * CELL, r * CELL), ((c + 1) * CELL - 1, (r + 1) * CELL - 1), (0, 0, 255), -1)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    for i in range(10):
        t = 2 if i % 3 == 0 else 1
        p = min(i * CELL, TARGET_SIZE - 1)
        cv2.line(img, (p, 0), (p, TARGET_SIZE - 1), (0, 160, 0), t)
        cv2.line(img, (0, p), (TARGET_SIZE - 1, p), (0, 160, 0), t)

    # Recognised digit in the top-left of each cell, so the printed digit stays visible next to it.
    for r in range(9):
        for c in range(9):
            d = int(res.grid[r, c])
            if d:
                cv2.putText(img, str(d), (c * CELL + 3, r * CELL + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1, cv2.LINE_AA)

    footer = np.full((36, TARGET_SIZE, 3), 255, np.uint8)
    if res.verdict == "invalid":
        text, colour = f"invalid: {len(res.conflicts)} conflicting cells", (0, 0, 255)
    else:
        text, colour = res.verdict, (0, 140, 0)
    cv2.putText(footer, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return np.vstack([img, footer])
