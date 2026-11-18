"""
pipeline.py
===========
映像全体の処理パイプライン本体 (process_video)。

動画入出力・カメラ行列/歪み補正マップの準備・フレームごとの検出/追跡呼び出し・
可視化ディスパッチ・診断ログ出力までを担う。CLI・設定読込は main.py に、
2D-3D対応追跡・姿勢推定ロジックは tracking.py に、診断ログの集約は
diagnostics.py に分離されている。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import tracking
from detector import build_detectors, detect_lines, describe_lines
from diagnostics import DiagnosticsLog
from matcher import match_lines
from writer import (
    MatchCSVWriter,
    draw_matches_side_by_side,
    draw_current_frame,
    draw_projection_only,
    draw_correspondence_only,
    draw_world_axes,
)


def process_video(cfg: dict):
    """映像全体を処理する。"""
    import random
    random.seed(cfg["io"].get("random_seed"))

    stage   = cfg.get("stage", 4)
    io_cfg  = cfg["io"]
    det_cfg = cfg["detection"]
    mat_cfg = cfg["matching"]
    proj_cfg = cfg["projection"]

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

    # 出力形式: "side_by_side"(既定, 前後フレーム並べ) / "current_only"(現在フレームのみ)
    # / "projection_only"(現在フレーム+投影線分のみ、他の注釈なし)
    # / "correspondence_only"(現在フレーム+姿勢推定に使われた対応線分のみ)
    # 文字列1つの代わりにリストを指定すると、1回の実行で複数のモードをそれぞれ
    # 別ファイルとして同時出力できる（diagログは全モード共通で全フォルダに保存）。
    _vis_mode_cfg = cfg["visualization"].get("mode", "side_by_side")
    vis_modes: list[str] = [_vis_mode_cfg] if isinstance(_vis_mode_cfg, str) else list(_vis_mode_cfg)
    multi_mode = len(vis_modes) > 1
    canvas_w_by_mode = {m: (out_w * 2 if m == "side_by_side" else out_w) for m in vis_modes}
    show_world_axes = bool(cfg["visualization"].get("show_world_axes", False))
    world_axes_length = float(cfg["visualization"].get("world_axes_length", 1.0))

    print(f"[INFO] 入力: {input_path}  ({w_orig}x{h_orig}, {fps:.1f}fps, {total}frames)")
    print(f"[INFO] 出力モード: {vis_modes}")
    print(f"[INFO] 出力解像度: { {m: f'{w}x{out_h}' for m, w in canvas_w_by_mode.items()} }")
    print(f"[INFO] 設定: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    # ── カメラ行列・歪み補正マップの準備 ────────────────────────────────
    K_mat = None
    undist_map1 = None
    undist_map2 = None
    undist_valid_mask = None

    if proj_cfg.get("enable", False) and stage >= 3:
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

        # 歪み補正はフレーム側で一度だけ行う（3D投影・マーカー検出は歪みなし
        # ピンホール前提になる）。変換マップは解像度分の1回だけ計算して使い回す。
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            K_mat, dist_arr, None, K_mat, img_size, cv2.CV_32FC1
        )
        # 実ソースが存在する画素=255 のマスク。BORDER_REPLICATE で滑らかに
        # 埋められる無効領域（実データが無く捏造された範囲）を検出線分の
        # クリップに使う（detect_lines の valid_mask 引数）。
        # 【注】BORDER_REPLICATEとマスククリップは別の問題への対策で重複しない:
        #   BORDER_REPLICATE: 黒/実画像の境界という「画像の枠そのもの」が
        #     強いエッジとして検出されるのを防ぐ（境界を無くす）
        #   マスククリップ: 本物のエッジが無効領域側へ誤って延長検出される
        #     のを防ぐ（検出後に捏造部分を切り捨てる）
        #   黒塗りのままクリップだけ行うと、境界線自体が有効領域側の線として
        #     検出される問題は防げない（実測で確認済み）。
        _mask_w, _mask_h = img_size
        undist_valid_mask = cv2.remap(
            np.full((_mask_h, _mask_w), 255, dtype=np.uint8),
            undist_map1, undist_map2, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        print("[INFO] フレーム歪み補正を有効化（cv2.remap、K・dist は camera 設定を使用）")
    else:
        img_size = (out_w, out_h)

    ctx = tracking.build_pipeline_context(cfg, K_mat, img_size)
    # ──────────────────────────────────────────────────────────────────────

    # json で指定されたパスの親ディレクトリが存在しない場合も作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    def _mode_output_path(mode: str) -> str:
        """複数モード出力時は <stem>_<mode><ext> に分ける。単一モード時は変更しない。"""
        if not multi_mode:
            return output_path
        p = Path(output_path)
        return str(p.with_name(f"{p.stem}_{mode}{p.suffix}"))

    # モードごとに動画ライターとフレーム画像保存先 (output/<stem>_frames/) を用意
    writers: dict[str, cv2.VideoWriter] = {}
    frames_dirs: dict[str, Path] = {}
    for _m in vis_modes:
        _out_m = _mode_output_path(_m)
        Path(_out_m).parent.mkdir(parents=True, exist_ok=True)
        writers[_m] = cv2.VideoWriter(_out_m, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                      (canvas_w_by_mode[_m], out_h))
        _fd = Path("output") / (Path(_out_m).stem + "_frames")
        _fd.mkdir(parents=True, exist_ok=True)
        frames_dirs[_m] = _fd

    csv_writer = MatchCSVWriter(csv_path)
    detector, descriptor, matcher = build_detectors(cfg)
    diag = DiagnosticsLog()

    # 診断ログ（pose_diag.csv等）の保存先は先頭モードのフォルダを基準に使う
    frames_dir = frames_dirs[vis_modes[0]]

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] 最初のフレームを読み込めません", file=sys.stderr)
        sys.exit(1)

    if resize_width > 0:
        prev_frame = cv2.resize(prev_frame, (out_w, out_h))
    if undist_map1 is not None:
        prev_frame = cv2.remap(prev_frame, undist_map1, undist_map2, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kl = detect_lines(prev_gray, detector, cfg, valid_mask=undist_valid_mask)
    prev_kl, prev_desc = describe_lines(prev_gray, prev_kl, descriptor) if stage >= 2 else (prev_kl, None)

    # 最初のフレームの姿勢・2D-3D対応を推定しておく（ループ内で引き継いで使う）
    pose_prev = tracking.acquire_pose(ctx, 0, prev_gray)
    prev_2d3d = tracking.seed_initial_2d3d(ctx, pose_prev, prev_kl)

    prev_state = tracking.TrackingState(
        frame=prev_frame, gray=prev_gray, kl=prev_kl, desc=prev_desc,
        pose=pose_prev, corr_2d3d=prev_2d3d,
    )

    frame_idx = 1
    total_matches = 0
    print("[INFO] 処理開始...")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if resize_width > 0:
            curr_frame = cv2.resize(curr_frame, (out_w, out_h))
        if undist_map1 is not None:
            curr_frame = cv2.remap(curr_frame, undist_map1, undist_map2, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_kl = detect_lines(curr_gray, detector, cfg, valid_mask=undist_valid_mask)
        curr_kl, curr_desc = describe_lines(curr_gray, curr_kl, descriptor) if stage >= 2 else (curr_kl, None)

        # ── Stage 2: LBD フレーム間対応 ──────────────────────────────
        if stage >= 2:
            matches = match_lines(prev_state.desc, curr_desc, matcher, ratio_thresh)
        else:
            matches = []
        total_matches += len(matches)

        csv_writer.write(frame_idx, prev_state.kl, curr_kl, matches)

        # ── 2D-3D対応追跡・姿勢推定本体（tracking.py） ─────────────
        result = tracking.process_frame(ctx, frame_idx, matches, prev_state, curr_kl, curr_gray)

        diag.log_corr(result.corr_diag_rows)
        diag.log_track(
            frame_idx, len(curr_kl), len(matches),
            result.tracked_kl_indices, result.geom_kl_indices, result.curr_2d3d,
        )

        for _m in vis_modes:
            if _m == "current_only":
                canvas = draw_current_frame(
                    curr_frame,
                    curr_kl,
                    cfg,
                    proj_lines=result.proj_curr,
                    tracked_kl_indices=result.tracked_kl_indices_vis,
                    geom_kl_indices=result.geom_kl_indices_vis,
                    stage=stage,
                )
            elif _m == "projection_only":
                canvas = draw_projection_only(curr_frame, result.proj_curr, cfg)
            elif _m == "correspondence_only":
                canvas = draw_correspondence_only(
                    curr_frame,
                    curr_kl,
                    cfg,
                    tracked_kl_indices=result.tracked_kl_indices_vis,
                    geom_kl_indices=result.geom_kl_indices_vis,
                    outlier_kl_indices=result.outlier_kl_indices,
                    line3d_labels=result.line3d_labels,
                )
            else:
                canvas = draw_matches_side_by_side(
                    prev_state.frame,
                    prev_state.kl,
                    curr_frame,
                    curr_kl,
                    matches,
                    cfg,
                    proj_lines1=result.proj_prev,
                    proj_lines2=result.proj_curr,
                    tracked_kl_indices1=result.tracked_prev_indices,
                    tracked_kl_indices2=result.tracked_kl_indices_vis,
                    geom_kl_indices2=result.geom_kl_indices_vis,
                    stage=stage,
                )

            # ── 世界座標軸の描画（精度の目視確認用。先行研究に合わせたもの）──
            if show_world_axes and K_mat is not None:
                if _m == "side_by_side":
                    if prev_state.pose is not None:
                        draw_world_axes(canvas, *prev_state.pose, K_mat, length=world_axes_length)
                    if result.pose_curr is not None:
                        draw_world_axes(canvas, *result.pose_curr, K_mat, length=world_axes_length, x_offset=out_w)
                elif result.pose_curr is not None:
                    draw_world_axes(canvas, *result.pose_curr, K_mat, length=world_axes_length)

            cw = canvas_w_by_mode[_m]
            if canvas.shape[1] != cw or canvas.shape[0] != out_h:
                print(f"[WARN] フレームサイズ不一致({_m}): {canvas.shape[1]}x{canvas.shape[0]} → {cw}x{out_h}")
                canvas = cv2.resize(canvas, (cw, out_h))
            writers[_m].write(canvas)

            frame_path = str(frames_dirs[_m] / f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, canvas)

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total - 1, 1) * 100
            print(
                f"  [{frame_idx:>5}/{total-1}] {pct:5.1f}%  "
                f"lines={len(prev_state.kl)}/{len(curr_kl)}  matches={len(matches)}"
            )

        diag.log_stability(frame_idx, prev_state.pose, result.pose_curr)

        prev_state = tracking.TrackingState(
            frame=curr_frame, gray=curr_gray, kl=curr_kl, desc=curr_desc,
            pose=result.pose_curr, corr_2d3d=result.curr_2d3d,
        )
        frame_idx += 1

    cap.release()
    for _w in writers.values():
        _w.release()
    csv_writer.close()

    diag.write_all(frames_dir)

    avg = total_matches / max(frame_idx - 1, 1)
    print(f"\n[DONE] 処理フレーム数: {frame_idx - 1}")
    print(f"       平均マッチ数/フレームペア: {avg:.1f}")
    for _m in vis_modes:
        print(f"       動画出力[{_m}] → {_mode_output_path(_m)}")
        print(f"       フレーム画像出力[{_m}] → {frames_dirs[_m]} フォルダ内")
    print(f"       CSV 出力 → {csv_path}")
