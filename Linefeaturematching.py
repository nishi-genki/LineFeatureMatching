"""
line_feature_matching.py
========================
映像の各フレームから線特徴を検出し、隣接フレーム間で対応付けを行うツール。

検出:    LSD (Line Segment Detector) + cv2.line_descriptor.LSDDetector
記述子:  LBD (Line Binary Descriptor)
対応付け: BinaryDescriptorMatcher (Hamming距離)
出力:    対応線を描画した動画 + 対応情報CSV

使い方:
    python line_feature_matching.py --input video.mp4
    python line_feature_matching.py --input video.mp4 --output result.mp4 --csv matches.csv
    python line_feature_matching.py --input video.mp4 --scale 0.5 --max_lines 100
"""

import argparse
import csv
import sys
from pathlib import Path
import os

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 初期化
# ─────────────────────────────────────────────


def build_detectors(scale: float = 0.8):
    """LSDDetector と BinaryDescriptor を生成する。"""
    lsd_params = cv2.line_descriptor.LSDParam()
    lsd_params.scale = scale
    detector = cv2.line_descriptor.LSDDetector_createLSDDetectorWithParams(lsd_params)
    descriptor = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
    matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
    return detector, descriptor, matcher


# ─────────────────────────────────────────────
# 検出・記述
# ─────────────────────────────────────────────


def detect_and_describe(gray: np.ndarray, detector, descriptor, max_lines: int = 200):
    """
    グレースケール画像から KeyLine と LBD 記述子を返す。

    Returns
    -------
    keylines : list[cv2.line_descriptor.KeyLine]
    descs    : np.ndarray  shape=(N, 32) dtype=uint8  (None if N==0)
    """
    # ─ 検出 ─
    keylines = detector.detect(gray, scale=2, numOctaves=1, mask=None)

    if not keylines:
        return [], None

    # スコア降順にソートして上位 max_lines だけ使う
    keylines = sorted(keylines, key=lambda k: k.response, reverse=True)[:max_lines]

    # octave を明示的に設定（LBD の要件）
    for kl in keylines:
        kl.octave = 0

    # ─ 記述 ─
    keylines, descs = descriptor.compute(gray, keylines)

    if descs is None or len(descs) == 0:
        return [], None

    return keylines, descs


# ─────────────────────────────────────────────
# マッチング
# ─────────────────────────────────────────────


def match_lines(descs1, descs2, matcher, ratio_thresh: float = 0.75):
    """
    Lowe's ratio test によるマッチング。

    Returns
    -------
    good_matches : list[cv2.DMatch]
    """
    if descs1 is None or descs2 is None:
        return []
    if len(descs1) < 2 or len(descs2) < 2:
        return []

    matches_knn = matcher.knnMatch(descs1, descs2, k=2)

    good = []
    for pair in matches_knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    return good


# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────

PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 180, 255),
    (255, 200, 40),
    (220, 80, 255),
    (40, 230, 200),
    (255, 140, 40),
    (140, 255, 40),
]


def draw_matches_side_by_side(
    frame1: np.ndarray, kl1, frame2: np.ndarray, kl2, matches, max_draw: int = 60
) -> np.ndarray:
    """
    2 フレームを横並びにして、左・右・対応線を別色で描画する。
    """
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:] = frame2

    color_left = (0, 255, 0)  # 左フレームの線分（緑）
    color_right = (255, 0, 0)  # 右フレームの線分（青）
    color_match = (0, 165, 255)  # 対応線（オレンジ）

    # 左フレームの線分
    for kl in kl1:
        pt1 = (int(kl.startPointX), int(kl.startPointY))
        pt2 = (int(kl.endPointX), int(kl.endPointY))
        cv2.line(canvas, pt1, pt2, color_left, 2)

    # 右フレームの線分
    for kl in kl2:
        pt1 = (int(kl.startPointX) + w1, int(kl.startPointY))
        pt2 = (int(kl.endPointX) + w1, int(kl.endPointY))
        cv2.line(canvas, pt1, pt2, color_right, 2)

    # 対応線（中点同士を結ぶ）
    for i, m in enumerate(matches[:max_draw]):
        kl_a = kl1[m.queryIdx]
        kl_b = kl2[m.trainIdx]
        pt1a = (int(kl_a.startPointX), int(kl_a.startPointY))
        pt2a = (int(kl_a.endPointX), int(kl_a.endPointY))
        pt1b = (int(kl_b.startPointX) + w1, int(kl_b.startPointY))
        pt2b = (int(kl_b.endPointX) + w1, int(kl_b.endPointY))
        mid_a = ((pt1a[0] + pt2a[0]) // 2, (pt1a[1] + pt2a[1]) // 2)
        mid_b = ((pt1b[0] + pt2b[0]) // 2, (pt1b[1] + pt2b[1]) // 2)
        cv2.line(canvas, mid_a, mid_b, color_match, 1, cv2.LINE_AA)

    # 情報テキスト
    info = f"Frame pair | Lines: {len(kl1)} / {len(kl2)} | Matches: {len(matches)}"
    cv2.putText(canvas, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def draw_lines_side_by_side(frame1: np.ndarray, kl1, frame2: np.ndarray, kl2, color=(0, 255, 0)) -> np.ndarray:
    """
    2フレームを横並びにして、検出した線分のみを描画する（マッチング線なし）。
    """
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:] = frame2

    # 左フレームの線
    for kl in kl1:
        pt1 = (int(kl.startPointX), int(kl.startPointY))
        pt2 = (int(kl.endPointX), int(kl.endPointY))
        cv2.line(canvas, pt1, pt2, color, 2)

    # 右フレームの線
    for kl in kl2:
        pt1 = (int(kl.startPointX) + w1, int(kl.startPointY))
        pt2 = (int(kl.endPointX) + w1, int(kl.endPointY))
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas


# ─────────────────────────────────────────────
# CSV ライター
# ─────────────────────────────────────────────


class MatchCSVWriter:
    HEADER = [
        "frame_idx",
        "match_rank",
        "distance",
        "kl1_sx",
        "kl1_sy",
        "kl1_ex",
        "kl1_ey",
        "kl1_angle",
        "kl1_length",
        "kl2_sx",
        "kl2_sy",
        "kl2_ex",
        "kl2_ey",
        "kl2_angle",
        "kl2_length",
    ]

    def __init__(self, path: str):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def write(self, frame_idx: int, kl1, kl2, matches):
        for rank, m in enumerate(matches):
            a = kl1[m.queryIdx]
            b = kl2[m.trainIdx]
            self._w.writerow(
                [
                    frame_idx,
                    rank,
                    m.distance,
                    f"{a.startPointX:.2f}",
                    f"{a.startPointY:.2f}",
                    f"{a.endPointX:.2f}",
                    f"{a.endPointY:.2f}",
                    f"{a.angle:.4f}",
                    f"{a.lineLength:.2f}",
                    f"{b.startPointX:.2f}",
                    f"{b.startPointY:.2f}",
                    f"{b.endPointX:.2f}",
                    f"{b.endPointY:.2f}",
                    f"{b.angle:.4f}",
                    f"{b.lineLength:.2f}",
                ]
            )

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────
# メインパイプライン
# ─────────────────────────────────────────────


def process_video(
    input_path: str,
    output_path: str,
    csv_path: str,
    max_lines: int = 200,
    ratio_thresh: float = 0.75,
    max_draw: int = 60,
    resize_width: int = 0,
):
    """映像全体を処理する。"""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 映像を開けません: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力サイズ（横並びなのでリサイズ後の幅 × 2）
    if resize_width > 0:
        scale_r = resize_width / w_orig
        out_w = resize_width
        out_h = int(h_orig * scale_r)
    else:
        out_w, out_h = w_orig, h_orig
    canvas_w = out_w * 2

    print(f"[INFO] 入力: {input_path}  ({w_orig}x{h_orig}, {fps:.1f}fps, {total}frames)")
    print(f"[INFO] 出力解像度: {canvas_w}x{out_h}")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, out_h),
    )

    csv_writer = MatchCSVWriter(csv_path)
    detector, descriptor, matcher = build_detectors()

    # フレーム画像出力用ディレクトリ
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ─ 最初のフレームを読み込み ─
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] 最初のフレームを読み込めません", file=sys.stderr)
        sys.exit(1)

    if resize_width > 0:
        prev_frame = cv2.resize(prev_frame, (out_w, out_h))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kl, prev_desc = detect_and_describe(prev_gray, detector, descriptor, max_lines)

    frame_idx = 1
    total_matches = 0

    print("[INFO] 処理開始...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if resize_width > 0:
            curr_frame = cv2.resize(curr_frame, (out_w, out_h))

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kl, curr_desc = detect_and_describe(curr_gray, detector, descriptor, max_lines)

        matches = match_lines(prev_desc, curr_desc, matcher, ratio_thresh)
        total_matches += len(matches)

        # CSV 書き込み
        csv_writer.write(frame_idx, prev_kl, curr_kl, matches)

        # 可視化フレームを動画に書き込み
        canvas = draw_matches_side_by_side(
            prev_frame,
            prev_kl,
            curr_frame,
            curr_kl,
            matches,
            max_draw,
        )

        # マッチング線なしの画像も保存
        canvas_lines = draw_lines_side_by_side(prev_frame, prev_kl, curr_frame, curr_kl, color=(0, 255, 0))

        # サイズ調整
        if canvas.shape[1] != canvas_w or canvas.shape[0] != out_h:
            print(f"[WARN] フレームサイズ不一致: {canvas.shape[1]}x{canvas.shape[0]} → {canvas_w}x{out_h}")
            canvas = cv2.resize(canvas, (canvas_w, out_h))
        writer.write(canvas)

        # フレーム画像として保存
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_path, canvas)

        # マッチング線なし画像も保存
        frame_path_lines = os.path.join(output_dir, f"frame_{frame_idx:05d}_lines.png")
        cv2.imwrite(frame_path_lines, canvas_lines)

        # 進捗表示
        if frame_idx % 30 == 0:
            pct = frame_idx / max(total - 1, 1) * 100
            print(
                f"  [{frame_idx:>5}/{total-1}] {pct:5.1f}%  "
                f"lines={len(prev_kl)}/{len(curr_kl)}  matches={len(matches)}"
            )

        # 次フレームへ
        prev_frame = curr_frame
        prev_gray = curr_gray
        prev_kl = curr_kl
        prev_desc = curr_desc
        frame_idx += 1

    cap.release()
    writer.release()
    csv_writer.close()

    avg = total_matches / max(frame_idx - 1, 1)
    print(f"\n[DONE] 処理フレーム数: {frame_idx - 1}")
    print(f"       平均マッチ数/フレームペア: {avg:.1f}")
    print(f"       動画出力 → {output_path}")
    print(f"       CSV 出力 → {csv_path}")
    print(f"       フレーム画像出力 → {output_dir} フォルダ内")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="映像から線特徴を検出しフレーム間対応付けを行う (LSD + LBD)")
    p.add_argument("--input", "-i", required=True, help="入力映像パス")
    p.add_argument("--output", "-o", default="", help="出力動画パス (省略時: <input>_line_match.mp4)")
    p.add_argument("--csv", "-c", default="", help="対応情報CSV (省略時: <input>_matches.csv)")
    p.add_argument("--max_lines", type=int, default=200, help="フレームあたりの最大検出線数 (default: 200)")
    p.add_argument("--ratio_thresh", type=float, default=0.75, help="Lowe ratio test 閾値 (default: 0.75)")
    p.add_argument("--max_draw", type=int, default=60, help="可視化で描画するマッチ数上限 (default: 60)")
    p.add_argument("--resize_width", type=int, default=0, help="リサイズ後の幅px (0=変更なし)")
    return p.parse_args()


def main():
    args = parse_args()
    stem = Path(args.input).stem

    output_path = args.output or f"{stem}_line_match.mp4"
    csv_path = args.csv or f"{stem}_matches.csv"

    process_video(
        input_path=args.input,
        output_path=output_path,
        csv_path=csv_path,
        max_lines=args.max_lines,
        ratio_thresh=args.ratio_thresh,
        max_draw=args.max_draw,
        resize_width=args.resize_width,
    )


if __name__ == "__main__":
    main()


def merge_keylines(keylines, angle_thresh=10.0, dist_thresh=50.0):
    """
    近接かつほぼ同じ角度の線分を統合する。
    angle_thresh: 統合する最大角度差（度）
    dist_thresh: 端点間の最大距離（ピクセル）
    """
    import math

    def angle_diff(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)

    merged = []
    used = [False] * len(keylines)

    for i, kl1 in enumerate(keylines):
        if used[i]:
            continue
        group = [kl1]
        used[i] = True
        for j, kl2 in enumerate(keylines):
            if i == j or used[j]:
                continue
            # 角度差
            a1 = math.degrees(kl1.angle)
            a2 = math.degrees(kl2.angle)
            if angle_diff(a1, a2) > angle_thresh:
                continue
            # 端点間距離（どちらかの端点同士が近ければOK）
            dists = [
                np.hypot(kl1.startPointX - kl2.startPointX, kl1.startPointY - kl2.startPointY),
                np.hypot(kl1.startPointX - kl2.endPointX, kl1.startPointY - kl2.endPointY),
                np.hypot(kl1.endPointX - kl2.startPointX, kl1.endPointY - kl2.startPointY),
                np.hypot(kl1.endPointX - kl2.endPointX, kl1.endPointY - kl2.endPointY),
            ]
            if min(dists) > dist_thresh:
                continue
            group.append(kl2)
            used[j] = True
        # グループ内の端点の最遠ペアで新しい線分を作る
        xs = [kl.startPointX for kl in group] + [kl.endPointX for kl in group]
        ys = [kl.startPointY for kl in group] + [kl.endPointY for kl in group]
        idx1 = np.argmin(xs)
        idx2 = np.argmax(xs)
        new_kl = group[0]  # 代表として最初のKeyLineをコピー
        new_kl.startPointX = xs[idx1]
        new_kl.startPointY = ys[idx1]
        new_kl.endPointX = xs[idx2]
        new_kl.endPointY = ys[idx2]
        merged.append(new_kl)
    return merged
