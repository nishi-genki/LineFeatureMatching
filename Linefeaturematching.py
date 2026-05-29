"""
line_feature_matching.py
========================
映像の各フレームから線特徴を検出し、隣接フレーム間で対応付けを行うツール。

検出:    LSD (Line Segment Detector) + cv2.line_descriptor.LSDDetector
記述子:  LBD (Line Binary Descriptor)
対応付け: BinaryDescriptorMatcher (Hamming距離)
出力:    対応線を描画した動画 + 対応情報CSV

使い方:
    python Linefeaturematching.py --config config.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 設定読み込み
# ─────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """
    JSON 設定ファイルを読み込み、デフォルト値で補完して返す。

    JSON の構造:
        {
            "io":            { "input", "output", "csv", "fallback_fps" },
            "detection":     { "detector", "max_lines", "resize_width",
                               "lsd_octave_scale", "lsd_num_octaves" },
            "descriptor":    { "lsd_scale", "edlines_min_length" },
            "matching":      { "ratio_thresh" },
            "visualization": { "max_draw", "only_matched", "line_thickness",
                               "match_thickness", "text_font_scale", "text_thickness" },
            "geometry":      { "vertical_epsilon" }
        }

    Parameters
    ----------
    config_path : str
        設定ファイルのパス。

    Returns
    -------
    dict
        確定済みパラメータ辞書（ネスト構造を保持）。
    """
    defaults = {
        "io": {
            "input": "",
            "output": "",
            "csv": "",
            "fallback_fps": 30.0,
        },
        "detection": {
            "detector": "lsd",
            "max_lines": 200,
            "resize_width": 0,
            "lsd_octave_scale": 2,
            "lsd_num_octaves": 1,
        },
        "descriptor": {
            "lsd_scale": 0.8,
            "edlines_min_length": 30.0,
        },
        "matching": {
            "ratio_thresh": 0.75,
        },
        "visualization": {
            "max_draw": 60,
            "only_matched": False,
            "line_thickness": 2,
            "match_thickness": 1,
            "text_font_scale": 0.6,
            "text_thickness": 1,
        },
        "geometry": {
            "vertical_epsilon": 1e-4,
        },
    }

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    # セクションごとにデフォルト値で補完
    cfg = {}
    for section, default_values in defaults.items():
        cfg[section] = {**default_values, **user_cfg.get(section, {})}

    # 必須チェック
    if not cfg["io"]["input"]:
        print("[ERROR] config.json の io.input を指定してください。", file=sys.stderr)
        sys.exit(1)

    # output / csv が空なら input のステム名から自動生成
    stem = Path(cfg["io"]["input"]).stem
    if not cfg["io"]["output"]:
        cfg["io"]["output"] = f"{stem}_line_match.mp4"
    if not cfg["io"]["csv"]:
        cfg["io"]["csv"] = f"{stem}_matches.csv"

    return cfg


# ─────────────────────────────────────────────
# 初期化
# ─────────────────────────────────────────────


def build_detectors(cfg: dict):
    """cfg の detection / descriptor をもとに検出器・記述子・マッチャーを生成する。"""
    detector_type = cfg["detection"]["detector"]

    if detector_type == "lsd":
        lsd_params = cv2.line_descriptor.LSDParam()
        lsd_params.scale = cfg["descriptor"]["lsd_scale"]
        detector = cv2.line_descriptor.LSDDetector_createLSDDetectorWithParams(lsd_params)
        descriptor = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
        return detector, descriptor, matcher
    elif detector_type == "edlines":
        descriptor = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
        return "edlines", descriptor, matcher
    else:
        raise ValueError("detector は 'lsd' または 'edlines' のみ指定可能です。")


# ─────────────────────────────────────────────
# 向き正規化
# ─────────────────────────────────────────────


def normalize_line_directions(keylines, cfg: dict):
    """線の向きを「左から右」（垂直なら「上から下」）に統一する。"""
    eps = cfg["geometry"]["vertical_epsilon"]
    for kl in keylines:
        if (kl.startPointX > kl.endPointX) or (
            abs(kl.startPointX - kl.endPointX) < eps and kl.startPointY > kl.endPointY
        ):
            kl.startPointX, kl.endPointX = kl.endPointX, kl.startPointX
            kl.startPointY, kl.endPointY = kl.endPointY, kl.startPointY
            kl.angle = kl.angle + np.pi if kl.angle < 0 else kl.angle - np.pi
    return keylines


# ─────────────────────────────────────────────
# 検出・記述
# ─────────────────────────────────────────────


def detect_and_describe(gray: np.ndarray, detector, descriptor, cfg: dict):
    """グレースケール画像から KeyLine と LBD 記述子を返す（向き正規化付き）。"""
    detector_type = cfg["detection"]["detector"]
    max_lines = cfg["detection"]["max_lines"]

    if detector_type == "lsd":
        return _detect_lsd(gray, detector, descriptor, max_lines, cfg)
    elif detector_type == "edlines":
        return _detect_edlines(gray, descriptor, max_lines, cfg["descriptor"]["edlines_min_length"], cfg)
    else:
        raise ValueError("detector は 'lsd' または 'edlines' のみ指定可能です。")


def _detect_lsd(gray, detector, descriptor, max_lines, cfg: dict):
    keylines = detector.detect(
        gray,
        scale=cfg["detection"]["lsd_octave_scale"],
        numOctaves=cfg["detection"]["lsd_num_octaves"],
        mask=None,
    )
    if not keylines:
        return [], None

    for kl in keylines:
        if not hasattr(kl, "octave") or kl.octave is None:
            kl.octave = 0

    keylines = sorted(keylines, key=lambda k: k.response, reverse=True)[:max_lines]
    keylines = normalize_line_directions(keylines, cfg)
    keylines, descs = descriptor.compute(gray, keylines)

    if descs is None or len(descs) == 0:
        return [], None
    return keylines, descs


def _detect_edlines(gray, descriptor, max_lines, min_length, cfg: dict):
    ed = cv2.ximgproc.createEdgeDrawing()
    ed.detectEdges(gray)
    lines = ed.detectLines()

    if lines is None or len(lines) == 0:
        return [], None

    candidates = []
    for l in lines:
        if isinstance(l[0], (np.ndarray, list)):
            x1, y1, x2, y2 = map(float, l[0])
        else:
            x1, y1, x2, y2 = map(float, l)
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            candidates.append((length, (x1, y1, x2, y2)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_lines = [c[1] for c in candidates[:max_lines]]

    keylines = []
    for idx, (x1, y1, x2, y2) in enumerate(selected_lines):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        kl = cv2.line_descriptor.KeyLine()
        kl.startPointX = x1
        kl.startPointY = y1
        kl.endPointX = x2
        kl.endPointY = y2
        kl.sPointInOctaveX = x1
        kl.sPointInOctaveY = y1
        kl.ePointInOctaveX = x2
        kl.ePointInOctaveY = y2
        kl.angle = np.arctan2(dy, dx)
        kl.lineLength = length
        kl.numOfPixels = int(length)
        kl.class_id = idx
        kl.octave = 0
        kl.response = length
        kl.pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        kl.size = length
        keylines.append(kl)

    if not keylines:
        return [], None

    keylines = normalize_line_directions(keylines, cfg)

    try:
        keylines, descs = descriptor.compute(gray, keylines)
    except cv2.error as e:
        print(f"[WARN] descriptor.compute failed: {e}")
        return [], None

    if descs is None or len(descs) == 0:
        return [], None
    return keylines, descs


# ─────────────────────────────────────────────
# マッチング
# ─────────────────────────────────────────────


def match_lines(descs1, descs2, matcher, ratio_thresh: float):
    """Lowe's ratio test によるマッチング。"""
    if descs1 is None or descs2 is None:
        return []
    if len(descs1) < 2 or len(descs2) < 2:
        return []

    good = []
    for pair in matcher.knnMatch(descs1, descs2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    return good


# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────


def draw_matches_side_by_side(frame1, kl1, frame2, kl2, matches, cfg: dict) -> np.ndarray:
    """2 フレームを横並びにして線分と対応線を描画する。"""
    vis_cfg = cfg["visualization"]
    max_draw = vis_cfg["max_draw"]
    only_matched = vis_cfg["only_matched"]
    line_thickness = vis_cfg["line_thickness"]
    match_thickness = vis_cfg["match_thickness"]
    text_font_scale = vis_cfg["text_font_scale"]
    text_thickness = vis_cfg["text_thickness"]

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:] = frame2

    color_left = (0, 255, 0)
    color_right = (255, 0, 0)
    color_match = (0, 165, 255)

    if only_matched:
        matched_idx1 = {m.queryIdx for m in matches[:max_draw]}
        matched_idx2 = {m.trainIdx for m in matches[:max_draw]}
        for i in matched_idx1:
            kl = kl1[i]
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color_left,
                line_thickness,
            )
        for i in matched_idx2:
            kl = kl2[i]
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color_right,
                line_thickness,
            )
    else:
        for kl in kl1:
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color_left,
                line_thickness,
            )
        for kl in kl2:
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color_right,
                line_thickness,
            )

    for m in matches[:max_draw]:
        kl_a, kl_b = kl1[m.queryIdx], kl2[m.trainIdx]
        mid_a = (
            (int(kl_a.startPointX) + int(kl_a.endPointX)) // 2,
            (int(kl_a.startPointY) + int(kl_a.endPointY)) // 2,
        )
        mid_b = (
            (int(kl_b.startPointX) + int(kl_b.endPointX)) // 2 + w1,
            (int(kl_b.startPointY) + int(kl_b.endPointY)) // 2,
        )
        cv2.line(canvas, mid_a, mid_b, color_match, match_thickness, cv2.LINE_AA)

    info = f"Frame pair | Lines: {len(kl1)} / {len(kl2)} | Matches: {len(matches)}"
    cv2.putText(
        canvas,
        info,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
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
            a, b = kl1[m.queryIdx], kl2[m.trainIdx]
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
# パイプライン
# ─────────────────────────────────────────────


def process_video(cfg: dict):
    """映像全体を処理する。"""
    io_cfg = cfg["io"]
    det_cfg = cfg["detection"]
    vis_cfg = cfg["visualization"]
    mat_cfg = cfg["matching"]

    input_path = io_cfg["input"]
    output_path = io_cfg["output"]
    csv_path = io_cfg["csv"]
    resize_width = det_cfg["resize_width"]
    max_draw = vis_cfg["max_draw"]
    only_matched = vis_cfg["only_matched"]
    ratio_thresh = mat_cfg["ratio_thresh"]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 映像を開けません: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or io_cfg["fallback_fps"]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_width > 0:
        scale_r = resize_width / w_orig
        out_w, out_h = resize_width, int(h_orig * scale_r)
    else:
        out_w, out_h = w_orig, h_orig
    canvas_w = out_w * 2

    print(f"[INFO] 入力: {input_path}  ({w_orig}x{h_orig}, {fps:.1f}fps, {total}frames)")
    print(f"[INFO] 出力解像度: {canvas_w}x{out_h}")
    print(f"[INFO] 設定: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, out_h))
    csv_writer = MatchCSVWriter(csv_path)
    detector, descriptor, matcher = build_detectors(cfg)

    output_dir = output_path.split(".")[0]
    os.makedirs(output_dir, exist_ok=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] 最初のフレームを読み込めません", file=sys.stderr)
        sys.exit(1)

    if resize_width > 0:
        prev_frame = cv2.resize(prev_frame, (out_w, out_h))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kl, prev_desc = detect_and_describe(prev_gray, detector, descriptor, cfg)

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
        curr_kl, curr_desc = detect_and_describe(curr_gray, detector, descriptor, cfg)

        matches = match_lines(prev_desc, curr_desc, matcher, ratio_thresh)
        total_matches += len(matches)

        csv_writer.write(frame_idx, prev_kl, curr_kl, matches)

        canvas = draw_matches_side_by_side(
            prev_frame,
            prev_kl,
            curr_frame,
            curr_kl,
            matches,
            cfg,
        )

        if canvas.shape[1] != canvas_w or canvas.shape[0] != out_h:
            print(f"[WARN] フレームサイズ不一致: {canvas.shape[1]}x{canvas.shape[0]} → {canvas_w}x{out_h}")
            canvas = cv2.resize(canvas, (canvas_w, out_h))
        writer.write(canvas)

        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_path, canvas)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total - 1, 1) * 100
            print(
                f"  [{frame_idx:>5}/{total-1}] {pct:5.1f}%  "
                f"lines={len(prev_kl)}/{len(curr_kl)}  matches={len(matches)}"
            )

        prev_frame, prev_gray = curr_frame, curr_gray
        prev_kl, prev_desc = curr_kl, curr_desc
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
    p = argparse.ArgumentParser(description="映像から線特徴を検出しフレーム間対応付けを行う (LSD + LBD or EDLines)")
    p.add_argument("--config", "-c", default="config.json", help="設定ファイルパス (default: config.json)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    process_video(cfg)


if __name__ == "__main__":
    main()
