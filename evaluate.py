"""Scores a results folder against ground-truth annotations with the Section 7 metrics.

    python evaluate.py --results results/ --annotations SudokuScan_devset/annotations
    python evaluate.py --results results/ --annotations ... --csv per_image.csv

Missing or unreadable outputs count as failures: no corners, and every cell wrong.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

TIME_LIMIT_S = 20.0
CONDITIONS = ("C1", "C2", "C3", "C4", "C5")


def side_length(quad: np.ndarray) -> float:
    """Mean length of the four sides of a TL, TR, BR, BL quadrilateral."""
    return float(np.mean(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)))


def read_grid_txt(path: Path) -> np.ndarray | None:
    try:
        lines = path.read_text().split()
    except OSError:
        return None
    if len(lines) != 9 or any(len(line) != 9 or not line.isdigit() for line in lines):
        return None
    return np.array([[int(ch) for ch in line] for line in lines])


def evaluate_image(ann: dict[str, Any], results: Path) -> dict[str, Any]:
    stem = Path(ann["image"]).stem
    gt_corners = np.asarray(ann["corners"], dtype=float)
    gt_grid = np.asarray(ann["grid"], dtype=int)

    try:
        res = json.loads((results / f"{stem}_result.json").read_text())
    except (OSError, ValueError):
        res = None

    runtime = res.get("runtime_s") if res else None
    row: dict[str, Any] = {
        "image": stem,
        "conditions": ann.get("conditions", []),
        "verdict": res.get("verdict", "?") if res else "missing",
        "runtime_s": runtime,
        "corner_err_pct": None,
        "robust": isinstance(runtime, (int, float)) and runtime <= TIME_LIMIT_S,
    }

    corners = res.get("corners") if res else None
    if corners is not None:
        err = np.linalg.norm(np.asarray(corners, dtype=float) - gt_corners, axis=1).mean()
        row["corner_err_pct"] = 100 * err / side_length(gt_corners)

    pred = read_grid_txt(results / f"{stem}_grid.txt")
    if pred is None:
        pred = np.full((9, 9), -1)  # no prediction: wrong everywhere, never "empty"
    row["cell_acc"] = float((pred == gt_grid).mean())
    row["exact"] = bool((pred == gt_grid).all())

    # Empty-cell F1 counts: the positive class is "empty".
    gt_empty, pred_empty = gt_grid == 0, pred == 0
    row["tp"] = int((gt_empty & pred_empty).sum())
    row["fp"] = int((~gt_empty & pred_empty).sum())
    row["fn"] = int((gt_empty & ~pred_empty).sum())

    row["confusions"] = Counter(
        (int(g), int(p)) for g, p in zip(gt_grid.ravel(), pred.ravel()) if g != p and p != -1
    )
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found = [r["corner_err_pct"] for r in rows if r["corner_err_pct"] is not None]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "n": len(rows),
        "found": len(found) / len(rows),
        "corner_mean": float(np.mean(found)) if found else None,
        "corner_median": float(np.median(found)) if found else None,
        "cell_acc": float(np.mean([r["cell_acc"] for r in rows])),
        "empty_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "exact": float(np.mean([r["exact"] for r in rows])),
        "robust": float(np.mean([r["robust"] for r in rows])),
    }


def _fmt(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def print_report(rows: list[dict[str, Any]], top_confusions: int) -> None:
    print(f"{'image':<14}{'cond':<12}{'verdict':<11}{'corner%':>8}{'cells':>8}{'exact':>7}{'time s':>8}")
    for r in rows:
        print(f"{r['image']:<14}{' '.join(r['conditions']):<12}{r['verdict']:<11}"
              f"{_fmt(r['corner_err_pct'], '.2f'):>8}{r['cell_acc']:>8.3f}{'yes' if r['exact'] else 'no':>7}"
              f"{_fmt(r['runtime_s'], '.2f'):>8}")

    print()
    print(f"{'group':<8}{'n':>4}{'found':>8}{'corner% mean':>14}{'median':>8}"
          f"{'cell acc':>10}{'empty F1':>10}{'exact':>8}{'robust':>8}")
    groups = [("all", rows)] + [(c, [r for r in rows if c in r["conditions"]]) for c in CONDITIONS]
    for name, group in groups:
        if not group:
            continue
        s = summarise(group)
        print(f"{name:<8}{s['n']:>4}{s['found']:>8.2f}{_fmt(s['corner_mean'], '.2f'):>14}"
              f"{_fmt(s['corner_median'], '.2f'):>8}{s['cell_acc']:>10.3f}{s['empty_f1']:>10.3f}"
              f"{s['exact']:>8.2f}{s['robust']:>8.2f}")

    confusions = sum((r["confusions"] for r in rows), Counter())
    if confusions and top_confusions:
        print("\nmost frequent confusions (true -> predicted, 0 = empty):")
        for (g, p), count in confusions.most_common(top_confusions):
            print(f"  {g} -> {p}: {count}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["image", "conditions", "verdict", "corner_err_pct", "cell_acc", "exact",
              "tp", "fp", "fn", "runtime_s", "robust"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "conditions": " ".join(r["conditions"])})


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SudokuScan results against annotations.")
    parser.add_argument("--results", type=Path, required=True, help="folder written by sudoku_scan.py")
    parser.add_argument("--annotations", type=Path, required=True, help="folder of ground-truth JSON files")
    parser.add_argument("--csv", type=Path, help="also write per-image metrics to this CSV file")
    parser.add_argument("--confusions", type=int, default=10, help="how many confusions to list")
    args = parser.parse_args()

    anns = [json.loads(p.read_text()) for p in sorted(args.annotations.glob("*.json"))]
    rows = [evaluate_image(a, args.results) for a in anns]
    if not rows:
        raise SystemExit(f"no annotations found in {args.annotations}")
    print_report(rows, args.confusions)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
