"""Parameter study for the Harris candidates: vary one setting at a time around the constants in corners.py.

    python tools/harris_sweep.py SudokuScan_devset/images SudokuScan_devset/annotations

For every setting, averaged over the images:
  candidates  - number of candidate points (fewer = easier corner selection)
  coverage    - fraction of images where each of the 4 true corners has a candidate within 1.5% of the side
  best err %  - distance from each true corner to its nearest candidate, as % of the side. This is the
                lowest corner error any selection rule could reach with these candidates.
  ms          - time for harris_candidates per image
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan import corners  # noqa: E402
from sudokuscan.preprocess import preprocess, to_work  # noqa: E402

VALUES = {
    "blockSize": [2, 3, 5, 7, 9, 11],
    "ksize": [3, 5, 7],
    "k": [0.02, 0.04, 0.06, 0.08, 0.12],
    "t": [0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
    "n": [5, 7, 8, 9, 11, 15, 21],
}


def load(images_dir, annotations_dir):
    """Preprocess every annotated image once; return (work image, true corners in work px, side in work px)."""
    data = []
    for ann_path in sorted(Path(annotations_dir).glob("*.json")):
        ann = json.loads(ann_path.read_text())
        image = cv2.imread(str(Path(images_dir) / ann["image"]))
        if image is None:
            continue
        prep = preprocess(image)
        quad = to_work(ann["corners"], prep["scale"])
        side = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1).mean()
        data.append((prep["work"], quad, side))
    return data


def measure(data, settings):
    counts, covered, errs, times = [], [], [], []
    for work, quad, side in data:
        start = time.perf_counter()
        _, cand = corners.harris_candidates(work, **settings)
        times.append(time.perf_counter() - start)
        counts.append(len(cand))
        if len(cand) == 0:
            covered.append(False)
            errs.append(np.nan)
            continue
        nearest = np.array([np.linalg.norm(cand - q, axis=1).min() for q in quad]) / side * 100
        covered.append(bool((nearest < 1.5).all()))
        errs.append(nearest.mean())
    return np.mean(counts), np.mean(covered), np.nanmean(errs), 1000 * np.mean(times)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = load(sys.argv[1], sys.argv[2])
    base = {"blockSize": corners.BLOCK_SIZE, "ksize": corners.KSIZE, "k": corners.K, "t": corners.T, "n": corners.N}
    print(f"{len(data)} images, base settings from corners.py: {base}\n")

    for name, values in VALUES.items():
        print(f"{name:>10}{'candidates':>12}{'coverage':>10}{'best err %':>12}{'ms':>7}")
        for v in values:
            n_cand, cov, err, ms = measure(data, {**base, name: v})
            mark = "  <- current" if v == base[name] else ""
            print(f"{v:>10}{n_cand:>12.0f}{cov:>10.2f}{err:>12.2f}{ms:>7.1f}{mark}")
        print()


if __name__ == "__main__":
    main()
