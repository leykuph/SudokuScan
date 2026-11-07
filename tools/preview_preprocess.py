"""Save photo, work and ink side by side for each image, to see what Stage 2 receives.

    python tools/preview_preprocess.py previews/ SudokuScan_devset/images/dev_20.jpg [more images ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sudokuscan.preprocess import preprocess  # noqa: E402

if len(sys.argv) < 3:
    sys.exit(__doc__)
outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)

for path in map(Path, sys.argv[2:]):
    image = cv2.imread(str(path))
    if image is None:
        print(path, "could not be read")
        continue
    prep = preprocess(image)
    small = cv2.resize(image, prep["work"].shape[::-1], interpolation=cv2.INTER_AREA)
    work = cv2.cvtColor(prep["work"], cv2.COLOR_GRAY2BGR)
    ink = cv2.cvtColor(255 - prep["ink"], cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(outdir / f"{path.stem}_stage1.jpg"), np.hstack([small, work, ink]))
    print(path.name, "->", outdir / f"{path.stem}_stage1.jpg")
