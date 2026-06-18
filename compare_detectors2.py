"""上限なしでの生検出数・線分長を比較。"""

import cv2
import numpy as np

VIDEO = "video/16.mp4"
MIN_LENGTH_EDLINES = 30
LSD_OCTAVE_SCALE = 2
LSD_NUM_OCTAVES = 1
LSD_SCALE = 0.8
LSD_DENSITY_TH = 0.5

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

ed_counts, lsd_counts = [], []
ed_all_lengths, lsd_all_lengths = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # EDLines（min_length フィルタのみ）
    ed.detectEdges(gray)
    lines_ed = ed.detectLines()
    ed_len = []
    if lines_ed is not None:
        for line in lines_ed:
            if isinstance(line[0], (np.ndarray, list)):
                x1, y1, x2, y2 = map(float, line[0])
            else:
                x1, y1, x2, y2 = map(float, line)
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= MIN_LENGTH_EDLINES:
                ed_len.append(length)
    ed_counts.append(len(ed_len))
    ed_all_lengths.extend(ed_len)

    # LSD（min_length なし）
    keylines_lsd = lsd.detect(gray, scale=LSD_OCTAVE_SCALE, numOctaves=LSD_NUM_OCTAVES)
    lsd_len = [kl.lineLength for kl in keylines_lsd] if keylines_lsd else []
    lsd_counts.append(len(lsd_len))
    lsd_all_lengths.extend(lsd_len)

cap.release()

for name, counts, lengths in [("EDLines", ed_counts, ed_all_lengths),
                               ("LSD",     lsd_counts, lsd_all_lengths)]:
    print(f"\n=== {name} (上限なし) ===")
    print(f"  検出線分数: 平均={np.mean(counts):.1f}, 中央値={np.median(counts):.1f}, 最小={np.min(counts)}, 最大={np.max(counts)}")
    if lengths:
        print(f"  線分長[px]: 平均={np.mean(lengths):.1f}, 中央値={np.median(lengths):.1f}, 最小={np.min(lengths):.1f}, 最大={np.max(lengths):.1f}")
        bins = [0, 30, 50, 100, 200, 500, 9999]
        labels = ["0-30", "30-50", "50-100", "100-200", "200-500", "500+"]
        for i, label in enumerate(labels):
            n = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
            print(f"    {label}px: {n} ({100*n/len(lengths):.1f}%)")
