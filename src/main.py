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
import time

import cv2
import numpy as np

from detector import build_detectors, detect_and_describe
from matcher import match_lines
from writer import MatchCSVWriter, draw_matches_side_by_side

# ─────────────────────────────────────────────
# 設定読み込み
# ─────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """
    JSON 設定ファイルを読み込み、default_config.json のデフォルト値で補完して返す。
    default_config.json は config_path と同じディレクトリに置く。
    """

    def deep_merge(a: dict, b: dict) -> dict:
        """b の値で a を再帰的に上書きして返す（非破壊）。"""
        out = {**a}
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    default_path = Path(config_path).parent / "default_config.json"
    with open(default_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    cfg = deep_merge(defaults, user_cfg)

    # 入力必須チェック
    if not cfg["io"].get("input"):
        print("[ERROR] config.json の io.input を指定してください。", file=sys.stderr)
        sys.exit(1)

    # 出力パスの自動生成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    stem = Path(cfg["io"]["input"]).stem
    if not cfg["io"].get("output"):
        cfg["io"]["output"] = str(output_dir / f"{stem}_line_match.mp4")
    if not cfg["io"].get("csv"):
        cfg["io"]["csv"] = str(output_dir / f"{stem}_matches.csv")

    return cfg


def store_config(cfg: dict, name: str = "config_used.json") -> Path:
    """
    読み込んだ cfg を出力先のファイル名（拡張子を除いたパス）下に保存して Path を返す。
    例: "output/edlines_parameter.mp4" -> "output/edlines_parameter/config_used.json"
    """
    out_file = Path(cfg["io"]["output"])
    out_dir = out_file.with_suffix("")  # 拡張子を除いたパスをディレクトリとして使う
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    try:
        out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 設定を保存 → {out_path}")
    except Exception as e:
        print(f"[WARN] 設定保存に失敗: {e}", file=sys.stderr)
    return out_path


# ─────────────────────────────────────────────
# パイプライン
# ─────────────────────────────────────────────


def process_video(cfg: dict):
    """映像全体を処理する。"""
    stage   = cfg.get("stage", 4)
    io_cfg  = cfg["io"]
    det_cfg = cfg["detection"]
    mat_cfg = cfg["matching"]
    proj_cfg = cfg["projection"]
    lm_cfg  = cfg["line_matching"]

    input_path = io_cfg["input"]
    output_path = io_cfg["output"]
    csv_path = io_cfg["csv"]
    resize_width = det_cfg["resize_width"]
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

    # ── 3D線分投影・線分姿勢推定の初期化 ────────────────────────────────
    projector = None
    pose_estimator = None
    line_matcher = None
    K_mat = None
    dist_arr = None
    min_line_corr = int(lm_cfg.get("min_lines", 3))

    if proj_cfg.get("enable", False) and stage >= 3:
        from projector import LineProjector, ProjectedLine, load_lines3d_csv
        from marker_pose import MarkerPoseEstimator
        from line_matcher import LineMatcher2D3D
        from line_pose import estimate_from_lines

        cam_cfg = cfg["camera"]
        fx, fy = cam_cfg["fx"], cam_cfg["fy"]
        cx, cy = cam_cfg["cx"], cam_cfg["cy"]

        # resize が有効な場合はカメラ行列をスケール
        if resize_width > 0:
            sr = resize_width / w_orig
            fx, fy, cx, cy = fx * sr, fy * sr, cx * sr, cy * sr
            img_size = (out_w, out_h)
        else:
            img_size = (cam_cfg["width"] or out_w, cam_cfg["height"] or out_h)

        K_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_arr = np.array(cam_cfg["dist"], dtype=np.float64)

        lines3d_path = proj_cfg.get("lines3d_csv", "")
        if not lines3d_path:
            print("[WARN] projection.lines3d_csv が未設定のため投影をスキップします。", file=sys.stderr)
        else:
            try:
                lines3d = load_lines3d_csv(lines3d_path)
                projector = LineProjector(fx, fy, cx, cy, cam_cfg["dist"], lines3d, img_size)
                print(f"[INFO] 3D線分を {len(lines3d)} 本読み込みました: {lines3d_path}")
            except Exception as e:
                print(f"[WARN] lines3d 読み込み失敗: {e}", file=sys.stderr)

        marker_jpg = proj_cfg.get("marker_jpg", "")
        marker_dat = proj_cfg.get("marker_dat", "")
        if projector and marker_jpg and marker_dat:
            try:
                pose_estimator = MarkerPoseEstimator(
                    marker_jpg,
                    marker_dat,
                    ratio=proj_cfg.get("marker_ratio", 0.75),
                    min_matches=proj_cfg.get("marker_min_matches", 8),
                )
                print(f"[INFO] マーカー姿勢推定を初期化: {marker_jpg}")
            except Exception as e:
                print(f"[WARN] MarkerPoseEstimator 初期化失敗: {e}", file=sys.stderr)

        if projector and pose_estimator:
            line_matcher = LineMatcher2D3D(
                angle_th_deg=float(lm_cfg.get("angle_th_deg", 10.0)),
                dist_th=float(lm_cfg.get("dist_th", 60.0)),
                overlap_th=float(lm_cfg.get("overlap_th", 0.2)),
            )
            print(
                f"[INFO] 線分マッチング (LineMatcher2D3D) を初期化: "
                f"angle={lm_cfg['angle_th_deg']}°, dist={lm_cfg['dist_th']}px, "
                f"overlap={lm_cfg['overlap_th']}"
            )
    # ──────────────────────────────────────────────────────────────────────

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

    # 最初のフレームの姿勢を推定しておく（ループ内でキャッシュして再利用）
    pose_prev = None
    if pose_estimator is not None:
        pose_prev = pose_estimator.estimate_pose(prev_gray, K_mat, dist_arr)

    # kl_idx → (p1_3d, p2_3d): 記述子マッチングで伝播する 2D-3D 対応テーブル
    prev_2d3d: dict[int, tuple] = {}
    if stage >= 4 and pose_prev is not None and line_matcher is not None:
        proj_init0 = projector.project_lines(*pose_prev)
        for proj, _kl, kl_idx in line_matcher.match(proj_init0, prev_kl):
            prev_2d3d[kl_idx] = (proj.p1_3d, proj.p2_3d)

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

        # ── Stage 2: LBD フレーム間対応 ──────────────────────────────
        if stage >= 2:
            matches = match_lines(prev_desc, curr_desc, matcher, ratio_thresh)
        else:
            matches = []
        total_matches += len(matches)

        csv_writer.write(frame_idx, prev_kl, curr_kl, matches)

        # ── Stage 4: LBD による 2D-3D 対応の伝播 ─────────────────────
        curr_2d3d: dict[int, tuple] = {}
        tracked_kl_indices: set[int] = set()
        if stage >= 4:
            for m in matches:
                if m.queryIdx in prev_2d3d:
                    curr_2d3d[m.trainIdx] = prev_2d3d[m.queryIdx]
                    tracked_kl_indices.add(m.trainIdx)

        # ── Stage 3/4: 3D投影 + 幾何マッチング + GOOPPnPL ────────────
        pose_curr: tuple | None = None
        proj_prev: list = []
        proj_curr: list = []

        if stage >= 3:
            # Step 1: マーカーから初期姿勢
            if pose_estimator is not None:
                pose_curr = pose_estimator.estimate_pose(curr_gray, K_mat, dist_arr)

            # Step 2: 幾何学的マッチングで curr_2d3d を補完
            ref_pose = pose_curr if pose_curr is not None else pose_prev
            if projector is not None and line_matcher is not None and ref_pose is not None:
                proj_init = projector.project_lines(*ref_pose)
                for proj, _kl, kl_idx in line_matcher.match(proj_init, curr_kl):
                    if kl_idx not in curr_2d3d:
                        curr_2d3d[kl_idx] = (proj.p1_3d, proj.p2_3d)

            # Step 3 (Stage 4 のみ): GOOPPnPL で姿勢精密化
            if stage >= 4:
                R_init_pose = pose_curr if pose_curr is not None else pose_prev
                if R_init_pose is not None and len(curr_2d3d) >= min_line_corr:
                    R_init, _ = R_init_pose
                    corr_for_pose = [
                        (ProjectedLine(p1_3d=p1, p2_3d=p2,
                                       pt1_2d=(curr_kl[idx].startPointX, curr_kl[idx].startPointY),
                                       pt2_2d=(curr_kl[idx].endPointX,   curr_kl[idx].endPointY)),
                         curr_kl[idx], idx)
                        for idx, (p1, p2) in curr_2d3d.items()
                    ]
                    try:
                        refined = estimate_from_lines(R_init, corr_for_pose, K_mat,
                                                      min_lines=min_line_corr)
                        if refined is not None:
                            pose_curr = refined
                    except Exception:
                        pass

            if pose_curr is None:
                pose_curr = pose_prev

            if projector is not None:
                if pose_prev is not None:
                    proj_prev = projector.project_lines(*pose_prev)
                if pose_curr is not None:
                    proj_curr = projector.project_lines(*pose_curr)

        canvas = draw_matches_side_by_side(
            prev_frame,
            prev_kl,
            curr_frame,
            curr_kl,
            matches,
            cfg,
            proj_lines1=proj_prev,
            proj_lines2=proj_curr,
            tracked_kl_indices2=tracked_kl_indices,
            stage=stage,
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
        pose_prev = pose_curr
        prev_2d3d = curr_2d3d
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

    # 読み込んだ設定を出力先フォルダへ保存
    store_config(cfg)

    start = time.perf_counter()
    process_video(cfg)
    elapsed = time.perf_counter() - start

    print(f"[TIME] 全処理時間: {elapsed:.3f}秒")


if __name__ == "__main__":
    main()
