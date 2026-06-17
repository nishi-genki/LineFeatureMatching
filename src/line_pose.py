"""
line_pose.py
============
2D-3D 線分対応から姿勢を推定する。
C++ GOPPnPL + ComputeCameraPose_l の Python 等価実装。

GOPPnPL のコスト関数 J を SO(3) 上で最小化する:
    J(R, t) = Σ ||(I - V_n)(R P_n + t)||² + Σ ||(I - V_n) R d_n||²

ここで V_n = I - n_n n_n^T (n_n: 2D 線の視線平面法線)。
並進 t は各 R に対して解析的に消去 (C++ calc_B, calc_t と同一)。
最適化は L-BFGS-B (scipy) による SO(3) 上のロドリゲス・パラメタ。
"""

from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _build_A(P: np.ndarray) -> np.ndarray:
    """
    R @ P = A @ vec(R) となる 3×9 行列を返す。
    vec(R) は Python 行優先 (R.flatten()) 規約。
    """
    A = np.zeros((3, 9), dtype=np.float64)
    A[0, 0:3] = P
    A[1, 3:6] = P
    A[2, 6:9] = P
    return A


def _build_V(
    p1: tuple[float, float],
    p2: tuple[float, float],
    cx: float, cy: float, f: float,
) -> Optional[np.ndarray]:
    """
    I - n n^T を返す (n: 2D 線が張る視線平面の法線)。
    C++ ComputeCameraPose_l の nvec 計算と同一。
    None を返す場合: 2 点が一致し法線が定まらない。
    """
    v1 = np.array([p1[0] - cx, p1[1] - cy, f], dtype=np.float64)
    v2 = np.array([p2[0] - cx, p2[1] - cy, f], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    nvec = np.cross(v1 / n1, v2 / n2)
    nlen = np.linalg.norm(nvec)
    if nlen < 1e-10:
        return None
    nvec /= nlen
    return np.eye(3) - np.outer(nvec, nvec)


def _calc_B(
    V_list: list[np.ndarray],
    A_list: list[np.ndarray],
) -> Optional[np.ndarray]:
    """
    並進の解析消去行列 B を計算する。C++ calc_B と同一。
    t = -B @ vec(R)
    """
    I3 = np.eye(3)
    sum_IminV   = sum(I3 - V for V in V_list)
    sum_IminV_A = sum((I3 - V) @ A for V, A in zip(V_list, A_list))
    try:
        return np.linalg.solve(sum_IminV, sum_IminV_A)   # (3, 9)
    except np.linalg.LinAlgError:
        return None


def _calc_D(
    V_list:  list[np.ndarray],
    A_list:  list[np.ndarray],
    B:       np.ndarray,
    Vd_list: list[np.ndarray],
    C_list:  list[np.ndarray],
    use_direction: bool,
) -> np.ndarray:
    """
    コスト行列 D を計算する。C++ calc_D と同一。
    J = vec(R)^T D vec(R)
    """
    I3 = np.eye(3)
    D1 = sum(
        (A - B).T @ (I3 - V) @ (A - B)
        for V, A in zip(V_list, A_list)
    )
    if use_direction and C_list:
        D2 = sum(
            C.T @ (I3 - Vd) @ C
            for Vd, C in zip(Vd_list, C_list)
        )
    else:
        D2 = np.zeros((9, 9), dtype=np.float64)
    return D1 + D2


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_from_lines(
    R_init:        np.ndarray,
    correspondences: list,
    K:             np.ndarray,
    min_lines:     int = 3,
    use_direction: bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    2D-3D 線分対応から姿勢 (R, t) を推定する。

    Parameters
    ----------
    R_init         : 3×3 初期回転行列 (マーカー姿勢などから取得)
    correspondences: [(ProjectedLine, KeyLine), ...] の対応リスト
                     line_matcher.LineMatcher2D3D.match() の出力
    K              : 3×3 カメラ内部行列
    min_lines      : 最低必要対応数
    use_direction  : True → 方向拘束も利用 (C++ use_direction=true と同一)

    Returns
    -------
    (R, t) または None (対応数不足・最適化失敗時)
    """
    cx, cy = float(K[0, 2]), float(K[1, 2])
    f = float(K[0, 0] + K[1, 1]) / 2.0

    V_list:  list[np.ndarray] = []
    A_list:  list[np.ndarray] = []
    Vd_list: list[np.ndarray] = []
    C_list:  list[np.ndarray] = []

    for proj, kl in correspondences:
        p1_2d = (float(kl.startPointX), float(kl.startPointY))
        p2_2d = (float(kl.endPointX),   float(kl.endPointY))

        V = _build_V(p1_2d, p2_2d, cx, cy, f)
        if V is None:
            continue

        P3d = np.asarray(proj.p1_3d, dtype=np.float64)
        d3d = np.asarray(proj.p2_3d, dtype=np.float64) - P3d

        V_list.append(V)
        A_list.append(_build_A(P3d))

        if use_direction:
            Vd_list.append(V)
            C_list.append(_build_A(d3d))

    if len(V_list) < min_lines:
        return None

    B = _calc_B(V_list, A_list)
    if B is None:
        return None

    D = _calc_D(V_list, A_list, B, Vd_list, C_list, use_direction)

    # ── SO(3) 上の最小化 (L-BFGS-B, ロドリゲスパラメタ) ──────────────────
    def objective(rv: np.ndarray) -> float:
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        r = R.flatten()
        return float(r @ D @ r)

    def gradient(rv: np.ndarray) -> np.ndarray:
        """
        dJ/drvec[k] = Σ_i (dJ/d vec(R)[i]) * jac[k,i]
        Python cv2.Rodrigues の jac 規約: shape (3, 9),  jac[k,i] = d vec(R)[i] / d rvec[k]
        """
        R, jac = cv2.Rodrigues(rv.reshape(3, 1))
        r = R.flatten()
        return (jac @ (2.0 * D @ r)).flatten()   # (3,9)@(9,) = (3,)

    rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
    res = minimize(
        objective,
        rvec_init.flatten(),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-15, "gtol": 1e-10},
    )

    R_opt, _ = cv2.Rodrigues(res.x.reshape(3, 1))
    t_opt    = -(B @ R_opt.flatten())   # C++ calc_t と同一

    return R_opt, t_opt.reshape(3)
