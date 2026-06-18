"""
line_pose.py
============
2D-3D 線分対応から姿勢を推定する。
GOOPPnPL pybind11 モジュールを使用。

GOOPPnPL_main の入力:
  Pl   : 3D 線分の始点・終点を交互に並べたリスト [p1_start, p1_end, p2_start, p2_end, ...]
  i_Pl : 2D 線分の始点・終点を交互に並べたリスト、各要素は (x-cx, y-cy, f)
  use_flag : [点, 線(位置), 線(方向)] を 0/1 で選択

戻り値: (R1, t1, R2, t2) — 2 候補解。コストの低い方を採用。
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

# GOOPPnPL pybind11 モジュールのインポート
_BIND_DIR = os.path.join(os.path.dirname(__file__),
                         "../GOOP-PnPL_pybind11/build")
sys.path.insert(0, os.path.abspath(_BIND_DIR))
try:
    import GOOPPnPL as _goppnpl
    _GOPPNPL_AVAILABLE = True
except ImportError:
    _GOPPNPL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# コスト計算（2 候補の選択に使用）
# ─────────────────────────────────────────────────────────────────────────────

def _build_V_line(p1_2d, p2_2d, cx, cy, f):
    v1 = np.array([p1_2d[0] - cx, p1_2d[1] - cy, f])
    v2 = np.array([p2_2d[0] - cx, p2_2d[1] - cy, f])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    nv = np.cross(v1 / n1, v2 / n2)
    nlen = np.linalg.norm(nv)
    if nlen < 1e-10:
        return None
    nv /= nlen
    return np.eye(3) - np.outer(nv, nv)


def _build_A(P):
    A = np.zeros((3, 9))
    A[0, 0:3] = P
    A[1, 3:6] = P
    A[2, 6:9] = P
    return A


def _calc_B(V_list, A_list):
    I3 = np.eye(3)
    sum_IminV   = sum(I3 - V for V in V_list)
    sum_IminV_A = sum((I3 - V) @ A for V, A in zip(V_list, A_list))
    try:
        return np.linalg.solve(sum_IminV, sum_IminV_A)
    except np.linalg.LinAlgError:
        return None


def _calc_D(V_list, A_list, B, Vd_list, C_list, use_direction):
    I3 = np.eye(3)
    D1 = sum((A - B).T @ (I3 - V) @ (A - B) for V, A in zip(V_list, A_list))
    if use_direction and C_list:
        D2 = sum(C.T @ (I3 - Vd) @ C for Vd, C in zip(Vd_list, C_list))
    else:
        D2 = np.zeros((9, 9))
    return D1 + D2


def _cost(R, t, V_list, A_list, B, Vd_list, C_list, use_direction):
    r = R.flatten()
    D = _calc_D(V_list, A_list, B, Vd_list, C_list, use_direction)
    return float(r @ D @ r)


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_from_lines(
    R_init:          np.ndarray,   # 不使用（後方互換のため残す）
    correspondences: list,
    K:               np.ndarray,
    min_lines:       int  = 3,
    use_direction:   bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    2D-3D 線分対応から姿勢 (R, t) を推定する。

    correspondences: [(ProjectedLine, KeyLine, kl_idx), ...] の対応リスト
    K              : 3×3 カメラ内部行列
    """
    if not _GOPPNPL_AVAILABLE:
        return None

    cx, cy = float(K[0, 2]), float(K[1, 2])
    f  = float(K[0, 0] + K[1, 1]) / 2.0

    Pl:   list[np.ndarray] = []   # 3D 端点 (始点, 終点, 始点, 終点, ...)
    i_Pl: list[np.ndarray] = []   # 2D 端点 (x-cx, y-cy, f) の形式

    V_list:  list[np.ndarray] = []
    A_list:  list[np.ndarray] = []
    Vd_list: list[np.ndarray] = []
    C_list:  list[np.ndarray] = []

    for item in correspondences:
        proj, kl = item[0], item[1]

        p1_3d = np.asarray(proj.p1_3d, dtype=np.float64)
        p2_3d = np.asarray(proj.p2_3d, dtype=np.float64)
        Pl.append(p1_3d)
        Pl.append(p2_3d)

        i_Pl.append(np.array([kl.startPointX - cx, kl.startPointY - cy, f]))
        i_Pl.append(np.array([kl.endPointX   - cx, kl.endPointY   - cy, f]))

        # コスト計算用（候補選択）
        V = _build_V_line(
            (kl.startPointX, kl.startPointY),
            (kl.endPointX,   kl.endPointY),
            cx, cy, f,
        )
        if V is not None:
            pm = (p1_3d + p2_3d) / 2.0   # 中点（GOOPPnPL_main と同じ基準点）
            V_list.append(V)
            A_list.append(_build_A(pm))
            if use_direction:
                Vd_list.append(V)
                C_list.append(_build_A(p2_3d - p1_3d))

    if len(Pl) // 2 < min_lines:
        return None

    use_flag = [0, 1, 1 if use_direction else 0]

    try:
        R1, t1, R2, t2 = _goppnpl.GOOPPnPL_main([], [], Pl, i_Pl, use_flag)
    except Exception:
        return None

    # 2 候補の選択: 3D 中点の平均 Z (カメラ座標系) が正の方が正解
    # Python の _cost 関数は V 行列の正規化方法が GOOPPnPL 内部と異なるため
    # コスト比較では誤った候補を選ぶことがある
    def _avg_z(R, t):
        R_ = np.asarray(R, dtype=np.float64)
        t_ = np.asarray(t, dtype=np.float64).flatten()
        zs = [float((R_ @ ((np.asarray(item[0].p1_3d) + np.asarray(item[0].p2_3d)) / 2.0) + t_)[2])
              for item in correspondences]
        return sum(zs) / max(len(zs), 1)

    if _avg_z(R2, t2) > _avg_z(R1, t1):
        return np.asarray(R2, dtype=np.float64), np.asarray(t2, dtype=np.float64).flatten()

    return np.asarray(R1, dtype=np.float64), np.asarray(t1, dtype=np.float64).flatten()
