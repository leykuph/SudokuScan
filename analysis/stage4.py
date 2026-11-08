import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def line_positions(path, axis):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    ink = 255 - img

    profile = ink.sum(axis=axis)

    peaks = []
    for p in range(0, 451, 50):
        lo = max(p - 20, 0)
        hi = min(p + 20, 450)
        peak = lo + np.argmax(profile[lo:hi])

        if (lo > 0 and peak == lo) or (hi < 450 and peak == hi - 1):
            peaks.append(np.nan)
        else:
            peaks.append(int(peak))

    return peaks

def straightness(img):

    left = img[:,:225]
    right = img[:,225:]

    prof_left = left.sum(axis=1)
    prof_right = right.sum(axis=1)

    peaksleft, peaksright = [], []
    for p in range(0, 451, 50):
        lo = max(p - 20, 0)
        hi = min(p + 20, 450)
        peakleft = lo + np.argmax(prof_left[lo:hi])
        peakright = lo + np.argmax(prof_right[lo:hi])
        peaksleft.append(int(peakleft))
        peaksright.append(int(peakright))
    return np.array(peaksright) - np.array(peaksleft)


def deviation(peaks):
    dev = np.array(peaks) - np.arange(0, 451, 50)
    mean = np.nanmean(np.abs(dev))
    max = np.nanmax(np.abs(dev))

    return max, mean

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

for model in ["H4", "A4", "A3"]:
    img = cv2.imread(f"results/dev_07_rect_{model}.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(model)
    ink = 255 - img
    print(model, "slant", straightness(ink))
    for axis in [1, 0]:
        peaks = line_positions(f"results/dev_07_rect_{model}.png", axis)
        print(model, "horizontal" if axis == 1 else "vertical", peaks, deviation(peaks))
