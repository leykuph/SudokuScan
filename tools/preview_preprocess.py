"""Save the Stage 1 outputs (greyscale, work, ink) side by side for inspection.

    python tools/preview_preprocess.py SudokuScan_devset/images/dev_20.jpg [more images ...] --outdir previews/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan.preprocess import preprocess  # noqa: E402


def labelled(img: np.ndarray, text: str) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--outdir", type=Path, default=Path("previews"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: could not read", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        prep = preprocess(image)
        dt = time.perf_counter() - t0
        small = cv2.resize(image, prep.work.shape[::-1], interpolation=cv2.INTER_AREA)
        panel = np.hstack([labelled(small, path.name), labelled(prep.work, "work (Harris input)"),
                           labelled(255 - prep.ink, "ink")])
        out = args.outdir / f"{path.stem}_stage1.jpg"
        cv2.imwrite(str(out), panel)
        print(f"{path.name}: {image.shape[1]}x{image.shape[0]} -> {prep.work.shape[1]}x{prep.work.shape[0]}, "
              f"scale {prep.scale:.3f}, {dt * 1000:.0f} ms -> {out}")


if __name__ == "__main__":
    main()
