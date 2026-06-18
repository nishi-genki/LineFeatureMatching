"""
EDLines と LSD の線分長分布ヒストグラムを生成する。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
rcParams["font.family"] = "Noto Sans CJK JP"

VIDEO = "video/16.mp4"
MIN_LENGTH_EDLINES = 30
LSD_OCTAVE_SCALE = 2
LSD_NUM_OCTAVES = 1
LSD_SCALE = 0.8
LSD_DENSITY_TH = 0.5
OUTPUT = "/home/nishi/デスクトップ/report/M1/06_19/figures/length_histogram.pdf"

cap = cv2.VideoCapture(VIDEO)

ed = cv2.ximgproc.createEdgeDrawing()
ed_params = ed.Params()
ed_params.LineFitErrorThreshold = 2.0
ed_params.MaxDistanceBetweenTwoLines = 10.0
ed.setParams(ed_params)

lsd_param = cv2.line_descriptor.LSDParam()
lsd_param.scale = LSD_SCALE
lsd_param.density_th = LSD_DENSITY_TH
lsd = cv2.line_descriptor.LSDDetector_createLSDDetectorWithParams(lsd_param)

ed_lengths, lsd_lengths = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ed.detectEdges(gray)
    lines_ed = ed.detectLines()
    if lines_ed is not None:
        for line in lines_ed:
            if isinstance(line[0], (np.ndarray, list)):
                x1, y1, x2, y2 = map(float, line[0])
            else:
                x1, y1, x2, y2 = map(float, line)
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= MIN_LENGTH_EDLINES:
                ed_lengths.append(length)

    keylines_lsd = lsd.detect(gray, scale=LSD_OCTAVE_SCALE, numOctaves=LSD_NUM_OCTAVES)
    if keylines_lsd:
        for kl in keylines_lsd:
            if kl.lineLength >= MIN_LENGTH_EDLINES:
                lsd_lengths.append(kl.lineLength)

cap.release()
print(f"EDLines: {len(ed_lengths)}本, LSD: {len(lsd_lengths)}本（30px以上）")

# ヒストグラム描画（並列）
bins = np.arange(30, 810, 30)  # 30px刻み, 30~800px

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, lengths, label, color in [
    (axes[0], ed_lengths, "EDLines", "steelblue"),
    (axes[1], lsd_lengths, "LSD",    "tomato"),
]:
    ax.hist(lengths, bins=bins, color=color, density=True)
    ax.set_title(label, fontsize=18)
    ax.set_xlabel("線分長 [px]", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim(30, 800)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

axes[0].set_ylabel("確率密度", fontsize=16)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight")
print(f"saved: {OUTPUT}")
