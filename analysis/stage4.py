import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
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

obl, a3, a4 = [], [], []

for path in sorted(Path(sys.argv[1]).glob("*_result.json")):
    r = json.loads(path.read_text())
    if r["corners"] is None:
        continue
    o = obliqueness(np.array(r["corners"]))
    print(path.stem, round(o, 2), round(r["a3_corner_err_px"], 1))

    obl.append(o)
    a3.append(r["a3_corner_err_px"])
    a4.append(r["a4_rms_px"])

plt.scatter(obl, a3, label="A3 fourth-corner error")
plt.scatter(obl, a4, label="A4 RMS error")
plt.xlabel("obliqueness (deg)")
plt.ylabel("error (px)")
plt.legend()
plt.savefig("residual_vs_obliqueness.png", dpi=200)
