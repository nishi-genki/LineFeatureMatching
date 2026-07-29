"""
test_marker_pose.py
====================
marker_pose.MarkerPoseEstimator を単体で試すためのスクリプト。
config.json の動画・カメラパラメータ・マーカー設定を使って、
指定フレームでのマーカー検出・姿勢推定結果を表示する。

使い方（リポジトリルートから実行。video/marker等のパスは config.json と同じくカレントディレクトリ基準）:
    python3 test_src/test_marker_pose.py               # 動画の1フレーム目
    python3 test_src/test_marker_pose.py --frame 10     # 10フレーム目
    python3 test_src/test_marker_pose.py --config config/config.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from marker_pose import MarkerPoseEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "config" / "config.json"))
    ap.add_argument("--frame", type=int, default=0, help="0始まりのフレーム番号")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    io_cfg   = cfg["io"]
    cam_cfg  = cfg["camera"]
    proj_cfg = cfg["projection"]

    K = np.array([
        [cam_cfg["fx"], 0,             cam_cfg["cx"]],
        [0,             cam_cfg["fy"], cam_cfg["cy"]],
        [0,             0,             1.0],
    ], dtype=np.float64)
    dist = np.array(cam_cfg["dist"], dtype=np.float64)

    marker_jpg = proj_cfg["marker_jpg"]
    marker_dat = proj_cfg["marker_dat"]
    print(f"[config] input={io_cfg['input']}")
    print(f"[config] marker_jpg={marker_jpg}")
    print(f"[config] marker_dat={marker_dat}")
    print(f"[config] K=\n{K}")
    print(f"[config] dist={dist}")

    estimator = MarkerPoseEstimator(
        marker_jpg, marker_dat,
        ratio=proj_cfg.get("marker_ratio", 0.75),
        min_matches=proj_cfg.get("marker_min_matches", 8),
    )
    mode = "ArUco" if estimator._aruco_detector is not None else "AKAZE(画像マーカー)"
    print(f"[estimator] 検出モード: {mode}"
          + (f" (id={estimator._aruco_id})" if estimator._aruco_detector is not None else ""))

    cap = cv2.VideoCapture(io_cfg["input"])
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {io_cfg['input']}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[ERROR] フレーム {args.frame} を読み込めません", file=sys.stderr)
        sys.exit(1)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"[frame] shape={frame.shape}")

    pose = estimator.estimate_pose(frame_gray, K, dist)
    if pose is None:
        print("[result] マーカーが検出できませんでした（None）")
        sys.exit(0)

    R, t = pose
    cam_center = -R.T @ t
    print("\n[result] 姿勢推定 成功")
    print(f"R =\n{R}")
    print(f"t = {t}")
    print(f"カメラ中心 (world) = {cam_center}")
    print(f"det(R) = {np.linalg.det(R):.6f}  (1.0に近いほど正常な回転行列)")

    # デバッグ用に検出結果を描画して保存
    vis = frame.copy()
    if estimator._aruco_detector is not None:
        corners, ids, _ = estimator._aruco_detector.detectMarkers(frame_gray)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    out_path = f"marker_pose_debug_frame{args.frame}.png"
    cv2.imwrite(out_path, vis)
    print(f"\n[debug] 検出結果を画像に保存: {out_path}")


if __name__ == "__main__":
    main()
