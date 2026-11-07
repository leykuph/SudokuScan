
import json
import sys
from pathlib import Path

import numpy as np


def obliqueness(corners):
    tl, tr, br, bl = corners

    leftDeg = getDeg(bl - tl)
    rightDeg = getDeg(br - tr)
    topDeg = getDeg(tr - tl)
    bottomDeg = getDeg(br - bl)

    diffLR = abs(leftDeg - rightDeg)
    diffTB = abs(topDeg - bottomDeg)

    return max(diffLR, diffTB)

def getDeg(d):
    arctan = np.arctan2(d[1], d[0])
    deg = np.degrees(arctan)
    return deg

for path in sorted(Path(sys.argv[1]).glob("*_result.json")):
    r = json.loads(path.read_text())
    if r["corners"] is None:
        continue
    o = obliqueness(np.array(r["corners"]))
    print(path.stem, round(o, 2), round(r["a3_corner_err_px"], 1))
