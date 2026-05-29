"""
main.py
=======
CLI エントリポイント・設定読み込み・メイン処理パイプライン。

使い方:
    python src/main.py --config config/config.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

from detector import build_detectors, detect_and_describe
from matcher import match_lines
from writer import MatchCSVWriter, draw_matches_side_by_side

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

    # output ディレクトリを確保
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # output / csv が空なら input のステム名から自動生成（output/ 配下）
    stem = Path(cfg["io"]["input"]).stem
    if not cfg["io"]["output"]:
        cfg["io"]["output"] = str(output_dir / f"{stem}_line_match.mp4")
    if not cfg["io"]["csv"]:
        cfg["io"]["csv"] = str(output_dir / f"{stem}_matches.csv")

    return cfg


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

    # json で指定されたパスの親ディレクトリが存在しない場合も作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, out_h))
    csv_writer = MatchCSVWriter(csv_path)
    detector, descriptor, matcher = build_detectors(cfg)

    # フレーム画像は output/<stem>_frames/ に保存
    frames_dir = Path("output") / (Path(output_path).stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

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

        frame_path = str(frames_dir / f"frame_{frame_idx:05d}.png")
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
    print(f"       フレーム画像出力 → {frames_dir} フォルダ内")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="映像から線特徴を検出しフレーム間対応付けを行う (LSD + LBD or EDLines)")
    p.add_argument(
        "--config",
        "-c",
        default="config/config.json",
        help="設定ファイルパス (default: config/config.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    process_video(cfg)


if __name__ == "__main__":
    main()
