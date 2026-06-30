"""
line_pose.py
============
2D-3D 線分対応から姿勢を推定する。
GOOPPnPL pybind11 モジュールを使用。

ロバスト推定: MAGSAC (scipy 不要)。
  - d=2 の残差分布で σ ∈ [0, σ_max] を積分消去した真の重み w_i = erfc(e_i / (√2·σ_max))
  - モデル選択は Σw_i を最大化
  - σ は重み付き中央値でデータから推定 → ユーザ指定のハード閾値不要

GOOPPnPL_main の入力:
  Pl   : 3D 線分の始点・終点を交互に並べたリスト [p1_start, p1_end, p2_start, p2_end, ...]
  i_Pl : 2D 線分の始点・終点を交互に並べたリスト、各要素は (x-cx, y-cy, f)
  use_flag : [点, 線(位置), 線(方向)] を 0/1 で選択

戻り値: (R1, t1, R2, t2) — 2 候補解。コストの低い方を採用。
"""

import math
import os
import random
import sys
from typing import Optional

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
# 内部: 3D線分投影
# ─────────────────────────────────────────────────────────────────────────────

def _project_line_2d(
    p1_3d: np.ndarray,
    p2_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """3D線分の端点をピンホールモデルで2Dに投影する（歪み補正なし）。"""
    def proj(P):
        Xc = R @ P + t
        if Xc[2] <= 0.0:
            return None
        u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
        v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
        return np.array([u, v], dtype=np.float64)
    return proj(p1_3d), proj(p2_3d)


# ─────────────────────────────────────────────────────────────────────────────
# 内部: MAGSAC++ （scipy 不要）
# ─────────────────────────────────────────────────────────────────────────────

# 角度事前フィルタ: これより大きい角度差の対応は残差 inf 扱い
_ANGLE_PREFILTER_DEG = 45.0


def _line_residual(proj, kl, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> float:
    """
    1対応の残差（検出線分端点の投影線分への平均垂線距離）。
    角度差が _ANGLE_PREFILTER_DEG を超える場合は inf を返す。
    """
    q1, q2 = _project_line_2d(
        np.asarray(proj.p1_3d, dtype=np.float64),
        np.asarray(proj.p2_3d, dtype=np.float64),
        R, t, K,
    )
    if q1 is None or q2 is None:
        return float('inf')

    pv = q2 - q1
    pv_len = float(np.linalg.norm(pv))
    if pv_len < 1e-3:
        return float('inf')
    pv_n = pv / pv_len

    d1 = np.array([kl.startPointX, kl.startPointY], dtype=np.float64)
    d2 = np.array([kl.endPointX,   kl.endPointY],   dtype=np.float64)
    dv = d2 - d1
    dv_len = float(np.linalg.norm(dv))
    if dv_len < 1e-3:
        return float('inf')

    cos_a = abs(float(np.dot(pv_n, dv / dv_len)))
    if math.acos(min(1.0, cos_a)) > math.radians(_ANGLE_PREFILTER_DEG):
        return float('inf')

    def perp(pt: np.ndarray) -> float:
        v = pt - q1
        return float(np.linalg.norm(v - float(np.dot(v, pv_n)) * pv_n))

    return (perp(d1) + perp(d2)) / 2.0


def _magsac_weight(e: float, sigma_max: float) -> float:
    """真のMAGSACソフト重み (d=2)。
    σ ∈ [0, sigma_max] を積分消去した解析解: w = erfc(e / (√2 · sigma_max))。
    e=0 で w=1、e→∞ で w→0。ハード閾値不要。
    """
    return math.erfc(e / (math.sqrt(2.0) * sigma_max))


def _compute_magsac_iters(n: int, sample_size: int = 3, confidence: float = 0.99) -> int:
    """インライア比率 0.5 を仮定した反復回数（最大 200）。"""
    p_sample = 0.5 ** sample_size
    k = math.log(1.0 - confidence) / math.log(max(1e-10, 1.0 - p_sample))
    return max(1, min(int(math.ceil(k)), 200))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """重み付き中央値を返す。"""
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    cumulative = 0.0
    for v, w in paired:
        cumulative += w
        if cumulative >= total / 2.0:
            return v
    return paired[-1][0]


def _magsac_inliers(
    correspondences: list,
    K: np.ndarray,
    min_lines: int,
    n_iter: int,
    sigma_max_px: float,
    use_direction: bool = True,
) -> list[int]:
    """
    MAGSAC (d=2) でインライアインデックスを返す。

    モデル選択: Σ erfc(e/√2/σ_max) を最大化（σを積分消去した真のMAGSAC重み）。
    σ 推定: 最良モデルの残差を重み付き中央値で推定（データ由来、ユーザ指定不要）。
    インライア: 3σ_est 以内（σ_est はデータから決定）。
    """
    n = len(correspondences)
    if n < min_lines:
        return list(range(n))

    best_score   = -1.0
    best_residuals: list[float] = []

    for _ in range(n_iter):
        sample_idx = random.sample(range(n), min_lines)
        sample = [correspondences[i] for i in sample_idx]

        pose = _estimate_pose_core(sample, K, min_lines, use_direction)
        if pose is None:
            continue
        R_s, t_s = pose

        residuals = [
            _line_residual(item[0], item[1], R_s, t_s, K)
            for item in correspondences
        ]

        score = sum(
            _magsac_weight(r, sigma_max_px)
            for r in residuals if r < float('inf')
        )

        if score > best_score:
            best_score     = score
            best_residuals = residuals

    if not best_residuals:
        return []

    # 重み付き中央値によるσ推定（データ由来、σ_maxに依存しない）
    finite_pairs = [
        (r, _magsac_weight(r, sigma_max_px))
        for r in best_residuals if r < float('inf')
    ]
    if not finite_pairs:
        return []

    finite_r, finite_w = zip(*finite_pairs)
    wmed = _weighted_median(list(finite_r), list(finite_w))
    sigma_est = max(wmed / 0.6745, 1.0)

    threshold = 3.0 * sigma_est
    inliers = [i for i, r in enumerate(best_residuals) if r <= threshold]
    return inliers if len(inliers) >= min_lines else []


def _ransac_inliers(
    correspondences: list,
    K: np.ndarray,
    min_lines: int,
    n_iter: int,
    inlier_thresh_px: float,
    use_direction: bool = True,
) -> list[int]:
    """
    RANSACでインライアインデックスを返す。

    モデル選択: inlier_thresh_px 以内のインライア数を最大化（ハード閾値）。
    """
    n = len(correspondences)
    if n < min_lines:
        return list(range(n))

    best_inliers: list[int] = []

    for _ in range(n_iter):
        sample_idx = random.sample(range(n), min_lines)
        sample = [correspondences[i] for i in sample_idx]

        pose = _estimate_pose_core(sample, K, min_lines, use_direction)
        if pose is None:
            continue
        R_s, t_s = pose

        inliers = [
            i for i, item in enumerate(correspondences)
            if _line_residual(item[0], item[1], R_s, t_s, K) <= inlier_thresh_px
        ]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return best_inliers if len(best_inliers) >= min_lines else []


# ─────────────────────────────────────────────────────────────────────────────
# 内部: 姿勢推定コア（ロバスト推定なし）
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_pose_core(
    correspondences: list,
    K: np.ndarray,
    min_lines: int = 3,
    use_direction: bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """GOOPPnPLで姿勢推定する（ロバスト推定なし）。"""
    if not _GOPPNPL_AVAILABLE:
        return None

    cx, cy = float(K[0, 2]), float(K[1, 2])
    f = float(K[0, 0] + K[1, 1]) / 2.0

    Pl:   list[np.ndarray] = []
    i_Pl: list[np.ndarray] = []

    for item in correspondences:
        proj, kl = item[0], item[1]
        p1_3d = np.asarray(proj.p1_3d, dtype=np.float64)
        p2_3d = np.asarray(proj.p2_3d, dtype=np.float64)
        Pl.append(p1_3d)
        Pl.append(p2_3d)
        i_Pl.append(np.array([kl.startPointX - cx, kl.startPointY - cy, f]))
        i_Pl.append(np.array([kl.endPointX   - cx, kl.endPointY   - cy, f]))

    if len(Pl) // 2 < min_lines:
        return None

    use_flag = [0, 1, 1 if use_direction else 0]

    try:
        R1, t1, R2, t2 = _goppnpl.GOOPPnPL_main([], [], Pl, i_Pl, use_flag)
    except Exception:
        return None

    R1 = np.asarray(R1, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64).flatten()
    R2 = np.asarray(R2, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64).flatten()

    def _total_residual(R, t):
        return sum(
            _line_residual(item[0], item[1], R, t, K)
            for item in correspondences
        )

    return (R2, t2) if _total_residual(R2, t2) < _total_residual(R1, t1) else (R1, t1)


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_from_lines(
    R_init:           np.ndarray,   # 不使用（後方互換のため残す）
    correspondences:  list,
    K:                np.ndarray,
    min_lines:        int   = 3,
    use_direction:    bool  = True,
    robust_method:    str   = "magsac",  # "magsac" / "ransac" / "none"
    sigma_max_px:     float = 20.0,
    ransac_thresh_px: float = 10.0,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    2D-3D 線分対応から姿勢 (R, t) を推定する。

    correspondences  : [(ProjectedLine, KeyLine, kl_idx), ...] の対応リスト
    K                : 3×3 カメラ内部行列
    robust_method    : ロバスト推定手法 ("magsac" / "ransac" / "none")
    sigma_max_px     : MAGSACのノイズスケール上限 [px]
    ransac_thresh_px : RANSACのインライア判定閾値 [px]
    """
    if not _GOPPNPL_AVAILABLE:
        return None
    if len(correspondences) < min_lines:
        return None

    if robust_method != "none" and len(correspondences) > min_lines:
        n_iter = _compute_magsac_iters(len(correspondences))
        if robust_method == "magsac":
            inlier_idx = _magsac_inliers(
                correspondences, K, min_lines, n_iter, sigma_max_px, use_direction,
            )
        else:  # ransac
            inlier_idx = _ransac_inliers(
                correspondences, K, min_lines, n_iter, ransac_thresh_px, use_direction,
            )
        if inlier_idx:
            correspondences = [correspondences[i] for i in inlier_idx]

    return _estimate_pose_core(correspondences, K, min_lines, use_direction)
