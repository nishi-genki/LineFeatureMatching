"""
main.py
=======
CLI エントリポイント・設定読み込み・メイン処理パイプライン。

使い方:
    python src/main.py --config config/config.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
import time

import cv2
import numpy as np

from detector import build_detectors, detect_lines, describe_lines
from matcher import match_lines, is_match_consistent
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

    # 出力パスの自動生成（動画は output/video/ にまとめる）
    output_dir = Path("output")
    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(cfg["io"]["input"]).stem
    if not cfg["io"].get("output"):
        cfg["io"]["output"] = str(video_dir / f"{stem}_line_match.mp4")
    if not cfg["io"].get("csv"):
        cfg["io"]["csv"] = str(output_dir / f"{stem}_matches.csv")

    return cfg


def load_rt_csv(path: str) -> dict[int, tuple]:
    """rt_results.csv を読み込み {frame_idx: (R, t)} を返す。success_flag==0 の行はスキップ。"""
    poses = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success_flag"]) == 0:
                continue
            frame = int(row["frame"])
            R = np.array([
                [float(row["R00"]), float(row["R01"]), float(row["R02"])],
                [float(row["R10"]), float(row["R11"]), float(row["R12"])],
                [float(row["R20"]), float(row["R21"]), float(row["R22"])],
            ], dtype=np.float64)
            t = np.array([float(row["t0"]), float(row["t1"]), float(row["t2"])], dtype=np.float64)
            poses[frame] = (R, t)
    return poses


def store_config(cfg: dict, name: str = "config_used.json") -> Path:
    """
    読み込んだ cfg をフレーム画像と同じフォルダに保存して Path を返す。
    例: "output/edlines_parameter.mp4" -> "output/edlines_parameter_frames/config_used.json"
    """
    out_file = Path(cfg["io"]["output"])
    out_dir = Path("output") / (out_file.stem + "_frames")  # フレーム保存先と同じフォルダ
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
    import random
    random.seed(cfg["io"].get("random_seed"))

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

    # 2D-3D 伝播時の 2D-2D 幾何整合フィルタ
    prop_cfg    = mat_cfg.get("propagation_filter", {})
    prop_enable = bool(prop_cfg.get("enable", False))
    prop_angle  = float(prop_cfg.get("angle_th_deg", 10.0))
    prop_dist   = float(prop_cfg.get("dist_th_px", 100.0))
    prop_ratio  = float(prop_cfg.get("length_ratio_th", 2.0))

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
    marker_init_frames = int(proj_cfg.get("marker_init_frames", -1))

    # 姿勢ジャンプゲート（0以下で無効）
    max_rot_jump_deg = float(lm_cfg.get("max_rot_jump_deg", -1.0))
    max_trans_jump   = float(lm_cfg.get("max_trans_jump", -1.0))

    if proj_cfg.get("enable", False) and stage >= 3:
        from projector import LineProjector, ProjectedLine, load_lines3d_csv
        from marker_pose import MarkerPoseEstimator
        from line_matcher import LineMatcher2D3D
        from line_pose import estimate_from_lines, _rotation_angle_deg

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

        # rt_csv からフレームごとの姿勢を読み込む
        rt_poses: dict[int, tuple] = {}
        rt_csv_path = proj_cfg.get("rt_csv", "")
        if rt_csv_path:
            try:
                rt_poses = load_rt_csv(rt_csv_path)
                print(f"[INFO] rt_csv 読み込み: {rt_csv_path}  ({len(rt_poses)} フレーム)")
            except Exception as e:
                print(f"[WARN] rt_csv 読み込み失敗: {e}", file=sys.stderr)

        def get_pose(frame_idx: int):
            """姿勢取得の優先順位: rt_csv > マーカー推定"""
            if frame_idx in rt_poses:
                return rt_poses[frame_idx]
            return None

    else:
        def get_pose(frame_idx: int):
            return None
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
    prev_kl = detect_lines(prev_gray, detector, cfg)
    prev_kl, prev_desc = describe_lines(prev_gray, prev_kl, descriptor) if stage >= 2 else (prev_kl, None)

    # 最初のフレームの姿勢を推定しておく（ループ内でキャッシュして再利用）
    pose_prev = get_pose(0)
    if pose_prev is None and pose_estimator is not None:
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
        curr_kl = detect_lines(curr_gray, detector, cfg)
        curr_kl, curr_desc = describe_lines(curr_gray, curr_kl, descriptor) if stage >= 2 else (curr_kl, None)

        # ── Stage 2: LBD フレーム間対応 ──────────────────────────────
        if stage >= 2:
            matches = match_lines(prev_desc, curr_desc, matcher, ratio_thresh)
        else:
            matches = []
        total_matches += len(matches)

        csv_writer.write(frame_idx, prev_kl, curr_kl, matches)

        # ── Stage 4: LBD による 2D-3D 対応の伝播 ─────────────────────
        # 伝播条件: (1) LBD 対応あり (2) 検出線分同士の 2D-2D 幾何整合
        #           （角度差・中点距離・長さ比。姿勢に依存しない）
        curr_2d3d: dict[int, tuple] = {}
        tracked_kl_indices: set[int] = set()
        tracked_prev_indices: set[int] = set()
        if stage >= 4:
            for m in matches:
                if m.queryIdx not in prev_2d3d:
                    continue
                if prop_enable and not is_match_consistent(
                        prev_kl[m.queryIdx], curr_kl[m.trainIdx],
                        prop_angle, prop_dist, prop_ratio):
                    continue
                curr_2d3d[m.trainIdx] = prev_2d3d[m.queryIdx]
                tracked_kl_indices.add(m.trainIdx)
                tracked_prev_indices.add(m.queryIdx)

        # ── Stage 3/4: 3D投影 + 幾何マッチング + GOOPPnPL ────────────
        pose_curr: tuple | None = None
        proj_prev: list = []
        proj_curr: list = []

        if stage >= 3:
            # Step 1: 姿勢取得（rt_csv > マーカー推定(初期フレームのみ) > 前フレーム引継ぎ）
            pose_curr = get_pose(frame_idx)
            use_marker = marker_init_frames < 0 or frame_idx < marker_init_frames
            if pose_curr is None and pose_estimator is not None and use_marker:
                pose_curr = pose_estimator.estimate_pose(curr_gray, K_mat, dist_arr)

            # Step 2: 幾何学的マッチングで curr_2d3d を補完（幾何由来のみ geom_2d3d に記録）
            geom_2d3d: dict[int, tuple] = {}
            ref_pose = pose_curr if pose_curr is not None else pose_prev
            if projector is not None and line_matcher is not None and ref_pose is not None:
                proj_init = projector.project_lines(*ref_pose)
                # LBD 引き継ぎ済みの 3D 線分は幾何マッチング対象から除外
                already_matched = {
                    (tuple(p1), tuple(p2)) for p1, p2 in curr_2d3d.values()
                }
                proj_init_filtered = [
                    pl for pl in proj_init
                    if (tuple(pl.p1_3d), tuple(pl.p2_3d)) not in already_matched
                ]
                for proj, _kl, kl_idx in line_matcher.match(proj_init_filtered, curr_kl):
                    geom_2d3d[kl_idx] = (proj.p1_3d, proj.p2_3d)
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
                                                      min_lines=min_line_corr,
                                                      sigma_max_px=float(lm_cfg.get("sigma_max_px", 20.0)),
                                                      ransac_thresh_px=float(lm_cfg.get("ransac_thresh_px", 10.0)),
                                                      robust_method=lm_cfg.get("robust_method", "magsac"),
                                                      y_down_constraint=bool(lm_cfg.get("y_down_constraint", True)))
                        # ── 姿勢ジャンプゲート ──────────────────────
                        # 前フレーム姿勢からの回転・カメラ位置の跳びが物理的に
                        # あり得ない大きさなら棄却する（反転解・破綻解の入口対策）。
                        if refined is not None and max_rot_jump_deg > 0:
                            R_new, t_new = refined
                            R_old, t_old = R_init_pose
                            if _rotation_angle_deg(R_new, R_old) > max_rot_jump_deg:
                                refined = None
                            elif max_trans_jump > 0:
                                c_new = -R_new.T @ np.asarray(t_new)
                                c_old = -R_old.T @ np.asarray(t_old)
                                if float(np.linalg.norm(c_new - c_old)) > max_trans_jump:
                                    refined = None
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

        geom_kl_indices = set(geom_2d3d.keys()) - tracked_kl_indices if stage >= 3 else set()
        canvas = draw_matches_side_by_side(
            prev_frame,
            prev_kl,
            curr_frame,
            curr_kl,
            matches,
            cfg,
            proj_lines1=proj_prev,
            proj_lines2=proj_curr,
            tracked_kl_indices1=tracked_prev_indices,
            tracked_kl_indices2=tracked_kl_indices,
            geom_kl_indices2=geom_kl_indices,
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

    # ── MAGSAC 診断ログの書き出し ─────────────────────────────────
    try:
        from line_pose import MAGSAC_DIAG
        if MAGSAC_DIAG:
            diag_path = frames_dir / "magsac_diag.csv"
            with open(diag_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(MAGSAC_DIAG[0].keys()))
                w.writeheader()
                w.writerows(MAGSAC_DIAG)
            print(f"[INFO] MAGSAC 診断ログ → {diag_path}  ({len(MAGSAC_DIAG)} 呼び出し)")
    except Exception as e:
        print(f"[WARN] MAGSAC 診断ログ出力失敗: {e}", file=sys.stderr)

    # ── GOOPPnPL 解診断ログの書き出し ─────────────────────────────
    try:
        from line_pose import POSE_DIAG
        if POSE_DIAG:
            diag_path = frames_dir / "pose_diag.csv"
            with open(diag_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(POSE_DIAG[0].keys()))
                w.writeheader()
                w.writerows(POSE_DIAG)
            n_calls = POSE_DIAG[-1]["call_id"] + 1
            print(f"[INFO] GOOPPnPL 解診断ログ → {diag_path}  ({n_calls} 呼び出し)")
    except Exception as e:
        print(f"[WARN] GOOPPnPL 解診断ログ出力失敗: {e}", file=sys.stderr)

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

    # 乱数シードを決定して cfg に埋め込む（config_used.json に保存される）
    # io.use_fixed_seed が true の場合は io.random_seed の値を使い、false の場合は時刻から生成する
    io_cfg_seed = cfg.get("io", {})
    if io_cfg_seed.get("use_fixed_seed", False):
        seed = io_cfg_seed.get("random_seed", 0)
    else:
        seed = int(time.time() * 1000) % (2 ** 32)
    cfg["io"]["random_seed"] = seed
    print(f"[INFO] random_seed: {seed} (fixed={io_cfg_seed.get('use_fixed_seed', False)})")

    # 読み込んだ設定を出力先フォルダへ保存
    store_config(cfg)

    start = time.perf_counter()
    process_video(cfg)
    elapsed = time.perf_counter() - start

    print(f"[TIME] 全処理時間: {elapsed:.3f}秒")


if __name__ == "__main__":
    main()
