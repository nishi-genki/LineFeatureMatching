"""
diagnostics.py
==============
2D-3D対応ログ・フレーム間追跡ログ・姿勢安定性ログを集約し、処理ループ終了後に
まとめて CSV へ書き出す。line_pose モジュールが内部で蓄積する MAGSAC/GOOPPnPL
の解診断ログもここで一緒に書き出す。
"""

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from line_pose import _rotation_angle_deg


@dataclass
class DiagnosticsLog:
    corr_diag: list[dict] = field(default_factory=list)
    track_diag: list[dict] = field(default_factory=list)
    stability_diag: list[dict] = field(default_factory=list)

    def log_corr(self, rows: list[dict]) -> None:
        self.corr_diag.extend(rows)

    def log_track(
        self,
        frame_idx: int,
        n_detected: int,
        n_lbd_matches: int,
        tracked_kl_indices: set[int],
        geom_kl_indices: set[int],
        curr_2d3d: dict,
    ) -> None:
        """フレーム間追跡ログ: 検出線分のうちLBD引き継ぎ/新規検出の内訳。"""
        self.track_diag.append({
            "frame_idx": frame_idx,
            "n_detected": n_detected,
            "n_lbd_matches": n_lbd_matches,
            "n_tracked_2d3d": len(tracked_kl_indices),
            "n_geom_2d3d_new": len(geom_kl_indices),
            "n_curr_2d3d": len(curr_2d3d),
        })

    def log_stability(self, frame_idx: int, pose_prev, pose_curr) -> None:
        """姿勢安定性ログ: 前フレームからのカメラ移動量・回転量。"""
        if pose_prev is None or pose_curr is None:
            return
        R_p, t_p = pose_prev
        R_c, t_c = pose_curr
        cam_p = -np.asarray(R_p, dtype=np.float64).T @ np.asarray(t_p, dtype=np.float64).flatten()
        cam_c = -np.asarray(R_c, dtype=np.float64).T @ np.asarray(t_c, dtype=np.float64).flatten()
        self.stability_diag.append({
            "frame_idx": frame_idx,
            "trans_delta": round(float(np.linalg.norm(cam_c - cam_p)), 5),
            "rot_delta_deg": round(_rotation_angle_deg(R_c, R_p), 4),
        })

    def write_all(self, frames_dir: Path) -> None:
        _write_csv(frames_dir / "corr_diag.csv", self.corr_diag, "2D-3D対応ログ")
        _write_csv(frames_dir / "track_diag.csv", self.track_diag, "フレーム間追跡ログ")
        _write_csv(frames_dir / "stability_diag.csv", self.stability_diag, "姿勢安定性ログ")
        _write_line_pose_diag(frames_dir)


def _write_csv(path: Path, rows: list[dict], label: str) -> None:
    if not rows:
        return
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] {label} → {path}  ({len(rows)} 件)")
    except Exception as e:
        print(f"[WARN] {label}出力失敗: {e}", file=sys.stderr)


def _write_line_pose_diag(frames_dir: Path) -> None:
    """line_pose モジュールが内部で蓄積する MAGSAC_DIAG / POSE_DIAG を書き出す。"""
    from line_pose import MAGSAC_DIAG, POSE_DIAG

    try:
        if MAGSAC_DIAG:
            path = frames_dir / "magsac_diag.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(MAGSAC_DIAG[0].keys()))
                w.writeheader()
                w.writerows(MAGSAC_DIAG)
            print(f"[INFO] MAGSAC 診断ログ → {path}  ({len(MAGSAC_DIAG)} 呼び出し)")
    except Exception as e:
        print(f"[WARN] MAGSAC 診断ログ出力失敗: {e}", file=sys.stderr)

    try:
        if POSE_DIAG:
            path = frames_dir / "pose_diag.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(POSE_DIAG[0].keys()))
                w.writeheader()
                w.writerows(POSE_DIAG)
            n_calls = POSE_DIAG[-1]["call_id"] + 1
            print(f"[INFO] GOOPPnPL 解診断ログ → {path}  ({n_calls} 呼び出し)")
    except Exception as e:
        print(f"[WARN] GOOPPnPL 解診断ログ出力失敗: {e}", file=sys.stderr)
