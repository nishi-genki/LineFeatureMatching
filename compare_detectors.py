"""
EDLines と LSD の検出線分特徴を定量的に比較するスクリプト。
各フレームの検出線分数・線分長（平均・最大・合計）を集計する。
"""

import cv2
import numpy as np

VIDEO = "video/16.mp4"
MAX_LINES = 200
MIN_LENGTH_EDLINES = 30
LSD_SCALE = 0.8
LSD_DENSITY_TH = 0.5
LSD_OCTAVE_SCALE = 2
LSD_NUM_OCTAVES = 1

cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"総フレーム数: {total_frames}")

# EDLines 検出器
ed = cv2.ximgproc.createEdgeDrawing()
ed_params = ed.Params()
ed_params.LineFitErrorThreshold = 2.0
ed_params.MaxDistanceBetweenTwoLines = 10.0
ed.setParams(ed_params)

# LSD 検出器
lsd_param = cv2.line_descriptor.LSDParam()
lsd_param.scale = LSD_SCALE
lsd_param.density_th = LSD_DENSITY_TH
lsd = cv2.line_descriptor.LSDDetector_createLSDDetectorWithParams(lsd_param)

stats = {"edlines": [], "lsd": []}

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- EDLines ---
    ed.detectEdges(gray)
    lines_ed = ed.detectLines()
    ed_lengths = []
    if lines_ed is not None:
        for line in lines_ed:
            if isinstance(line[0], (np.ndarray, list)):
                x1, y1, x2, y2 = map(float, line[0])
            else:
                x1, y1, x2, y2 = map(float, line)
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= MIN_LENGTH_EDLINES:
                ed_lengths.append(length)
        ed_lengths.sort(reverse=True)
        ed_lengths = ed_lengths[:MAX_LINES]
    stats["edlines"].append(ed_lengths)

    # --- LSD ---
    keylines_lsd = lsd.detect(gray, scale=LSD_OCTAVE_SCALE, numOctaves=LSD_NUM_OCTAVES)
    lsd_lengths = []
    if keylines_lsd:
        for kl in keylines_lsd:
            lsd_lengths.append(kl.lineLength)
        lsd_lengths.sort(reverse=True)
        lsd_lengths = lsd_lengths[:MAX_LINES]
    stats["lsd"].append(lsd_lengths)

    frame_idx += 1

cap.release()
print(f"処理フレーム数: {frame_idx}")

# 統計集計
for method, all_lengths in stats.items():
    counts = [len(ls) for ls in all_lengths]
    all_flat = [l for ls in all_lengths for l in ls]
    print(f"\n=== {method.upper()} ===")
    print(f"  検出線分数: 平均={np.mean(counts):.1f}, 中央値={np.median(counts):.1f}, 最小={np.min(counts)}, 最大={np.max(counts)}")
    if all_flat:
        print(f"  線分長[px]: 平均={np.mean(all_flat):.1f}, 中央値={np.median(all_flat):.1f}, 最小={np.min(all_flat):.1f}, 最大={np.max(all_flat):.1f}")
    # 長さ分布（ビン別）
    bins = [30, 50, 100, 200, 500, 9999]
    labels = ["30-50", "50-100", "100-200", "200-500", "500+"]
    for i, label in enumerate(labels):
        n = sum(1 for l in all_flat if bins[i] <= l < bins[i+1])
        print(f"    {label}px: {n} ({100*n/len(all_flat):.1f}%)")
