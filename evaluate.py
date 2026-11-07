"""Section 7 metrics for a results folder, against ground-truth annotations.

    python evaluate.py results/ SudokuScan_devset/annotations
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def read_outputs(results, stem):
    """The result JSON ({} if missing) and the 9x9 grid from grid.txt (all -1, i.e. all wrong, if missing)."""
    try:
        res = json.loads((results / f"{stem}_result.json").read_text())
    except (OSError, ValueError):
        res = {}
    try:
        grid = np.array([[int(c) for c in line] for line in (results / f"{stem}_grid.txt").read_text().split()])
        assert grid.shape == (9, 9)
    except (OSError, ValueError, AssertionError):
        grid = np.full((9, 9), -1)
    return res, grid


def score_image(results, ann) -> dict[str, Any]:
    stem = Path(ann["image"]).stem
    res, pred = read_outputs(results, stem)
    gt = np.array(ann["grid"])
    quad = np.array(ann["corners"], dtype=float)

    corner_err = None
    if res.get("corners") is not None:
        side = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean()   # mean of the 4 sides
        corner_err = 100 * np.linalg.norm(np.array(res["corners"]) - quad, axis=1).mean() / side

    return {
        "image": stem,
        "conditions": ann.get("conditions", []),
        "verdict": res.get("verdict", "missing"),
        "corner_err": corner_err,                          # % of the true side length
        "cell_acc": (pred == gt).mean(),
        "exact": (pred == gt).all(),
        "robust": res.get("runtime_s", 999) <= 20,         # an output file exists and within the time limit
        "tp": ((gt == 0) & (pred == 0)).sum(),             # empty-cell F1: "empty" is the positive class
        "fp": ((gt != 0) & (pred == 0)).sum(),
        "fn": ((gt == 0) & (pred != 0)).sum(),
        "confusions": Counter((int(g), int(p)) for g, p in zip(gt.flat, pred.flat) if g != p and p != -1),
    }


def summary_line(name, rows):
    errs = [r["corner_err"] for r in rows if r["corner_err"] is not None]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    corner = f"{np.mean(errs):15.2f}" if errs else f"{'-':>15}"
    return (f"{name:<6}{len(rows):>4}{len(errs) / len(rows):>7.2f}{corner}"
            f"{np.mean([r['cell_acc'] for r in rows]):>10.3f}{f1:>10.3f}"
            f"{np.mean([r['exact'] for r in rows]):>8.2f}{np.mean([r['robust'] for r in rows]):>8.2f}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    results, annotations = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [score_image(results, json.loads(p.read_text())) for p in sorted(annotations.glob("*.json"))]

    print(f"{'image':<10}{'conditions':<12}{'verdict':<11}{'corner %':>9}{'cells':>7}{'exact':>7}")
    for r in rows:
        corner = f"{r['corner_err']:9.2f}" if r["corner_err"] is not None else f"{'-':>9}"
        print(f"{r['image']:<10}{' '.join(r['conditions']):<12}{r['verdict']:<11}{corner}"
              f"{r['cell_acc']:>7.3f}{'yes' if r['exact'] else 'no':>7}")

    print(f"\n{'group':<6}{'n':>4}{'found':>7}{'corner % mean':>15}{'cell acc':>10}{'empty F1':>10}{'exact':>8}{'robust':>8}")
    print(summary_line("all", rows))
    for c in ("C1", "C2", "C3", "C4", "C5"):
        group = [r for r in rows if c in r["conditions"]]
        if group:
            print(summary_line(c, group))

    confusions = sum((r["confusions"] for r in rows), Counter())
    print("\nmost frequent confusions (true -> predicted, 0 = empty):")
    for (g, p), n in confusions.most_common(10):
        print(f"  {g} -> {p}: {n}")


if __name__ == "__main__":
    main()
