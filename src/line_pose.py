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

解の選択: GOOPPnPL_all の全候補（J昇順）から
  実数解 & カメラ前方 & カメラY軸下向き(r12<0) & 再投影残差有限
を満たす J 最小の候補を採用し OIPnPL で改良する（鏡映反転解対策）。
フィルタ全滅時は上位2解から参照姿勢に近い方（無ければ残差の小さい方）。
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

# MAGSAC 診断ログ（呼び出しごとに σ 推定の内訳を蓄積。main.py が CSV に出力する）
MAGSAC_DIAG: list[dict] = []

# GOOPPnPL 解診断ログ（最終姿勢推定ごとに全候補解を蓄積。main.py が CSV に出力する）
# 1行 = 1候補解。kind="refined" は OIPnPL 改良済みの上位2解（実際の選択対象）、
# kind="cand" は大域解法の生の候補解（J 昇順）。
POSE_DIAG: list[dict] = []


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
    R_ref: "Optional[np.ndarray]" = None,
    y_down_constraint: bool = True,
) -> list[int]:
    """
    MAGSAC (d=2) でインライアインデックスを返す。

    モデル選択: Σ erfc(e/√2/σ_max) を最大化（σを積分消去した真のMAGSAC重み）。
    LO ステップ: 最良モデルの緩いインライア集合で再フィットし残差を再計算
                 （最小サンプルの過学習残差によるσの過小推定を防ぐ）。
    σ 推定: 再フィット後の残差を重み付き中央値で推定（データ由来、ユーザ指定不要）。
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

        pose = _estimate_pose_core(sample, K, min_lines, use_direction, R_ref,
                                   y_down_constraint)
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

    # ── LO (σ-consensus) ステップ ────────────────────────────────
    # 最小サンプルフィットの残差はサンプル自身に過学習して ≈0 となり、
    # σを過小推定する。緩いインライア集合（residual ≤ σ_max）で再フィットし、
    # 非最小フィットの正直な残差に置き換える。スコアが改善する間繰り返す。
    lo_applied = 0
    for _ in range(3):
        lo_idx = [i for i, r in enumerate(best_residuals) if r <= sigma_max_px]
        if len(lo_idx) < min_lines:
            break
        lo_pose = _estimate_pose_core(
            [correspondences[i] for i in lo_idx], K, min_lines, use_direction,
            R_ref, y_down_constraint,
        )
        if lo_pose is None:
            break
        R_lo, t_lo = lo_pose
        lo_residuals = [
            _line_residual(item[0], item[1], R_lo, t_lo, K)
            for item in correspondences
        ]
        lo_score = sum(
            _magsac_weight(r, sigma_max_px)
            for r in lo_residuals if r < float('inf')
        )
        if lo_score <= best_score:
            break
        best_score     = lo_score
        best_residuals = lo_residuals
        lo_applied += 1

    # 重み付き中央値によるσ推定
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

    sorted_r = sorted(finite_r)
    MAGSAC_DIAG.append({
        "n_corr":         n,
        "n_finite":       len(sorted_r),
        "res_median":     sorted_r[len(sorted_r) // 2],
        "res_p90":        sorted_r[int(len(sorted_r) * 0.9)] if len(sorted_r) > 1 else sorted_r[-1],
        "res_max":        sorted_r[-1],
        "wmed":           wmed,
        "sigma_est":      sigma_est,
        "threshold":      threshold,
        "n_inliers":      len(inliers),
        "n_inliers_gt15": sum(1 for i in inliers if best_residuals[i] > 15.0),
        "lo_applied":     lo_applied,
    })

    return inliers if len(inliers) >= min_lines else []


def _ransac_inliers(
    correspondences: list,
    K: np.ndarray,
    min_lines: int,
    n_iter: int,
    inlier_thresh_px: float,
    use_direction: bool = True,
    R_ref: "Optional[np.ndarray]" = None,
    y_down_constraint: bool = True,
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

        pose = _estimate_pose_core(sample, K, min_lines, use_direction, R_ref,
                                   y_down_constraint)
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

def _rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """2つの回転行列の相対回転角 [deg] を返す。"""
    cos_a = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))


def _append_pose_diag(all_sols, R1, t1, R2, t2, res1, res2,
                      selected, correspondences, K, R_ref):
    """GOOPPnPL の全候補解を POSE_DIAG に追記する。

    kind="refined": OIPnPL 改良済みの上位2解。idx は 1/2。
    kind="cand"   : 大域解法の生の候補解（J 昇順）。複素解 (is_real=False) の
                    R/t は実部のみで構成されるため幾何的な意味を持たない。
    cam_x/y/z     : カメラ中心 -Rᵀt（鏡映解は原点対称に現れるため確認用）
    selected      : ("refined", 1/2) または ("cand", idx)。採用された行に True が付く
    """
    call_id = POSE_DIAG[-1]["call_id"] + 1 if POSE_DIAG else 0
    n_corr = len(correspondences)
    if selected[0] == "refined":
        R_sel = R1 if selected[1] == 1 else R2
    else:
        R_sel = np.asarray(all_sols[selected[1]][0], dtype=np.float64)

    def _row(kind, idx, R, t, J, is_real, is_front, residual):
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).flatten()
        c = -R.T @ t
        return {
            "call_id":     call_id,
            "n_corr":      n_corr,
            "kind":        kind,
            "idx":         idx,
            "J":           J,
            "is_real":     is_real,
            "is_front":    is_front,
            "residual":    residual,
            "rot_vs_ref":  _rotation_angle_deg(R, R_ref) if R_ref is not None else float('nan'),
            "rot_vs_sel":  _rotation_angle_deg(R, R_sel),
            "cam_x":       float(c[0]),
            "cam_y":       float(c[1]),
            "cam_z":       float(c[2]),
            # カメラY軸のワールドZ成分。「カメラYが下向き ⇔ R[1,2]<0」仮説の検証用
            "r12":         float(R[1, 2]),
            "selected":    ((kind, idx) == selected),
        }

    POSE_DIAG.append(_row("refined", 1, R1, t1, float('nan'), True, None, res1))
    POSE_DIAG.append(_row("refined", 2, R2, t2, float('nan'), True, None, res2))

    for i, (R, t, J, is_real, is_front) in enumerate(all_sols):
        residual = float('nan')
        if is_real:
            residual = sum(
                _line_residual(item[0], item[1],
                               np.asarray(R, dtype=np.float64),
                               np.asarray(t, dtype=np.float64).flatten(), K)
                for item in correspondences
            )
        POSE_DIAG.append(_row("cand", i, R, t, J, is_real, is_front, residual))


def _estimate_pose_core(
    correspondences: list,
    K: np.ndarray,
    min_lines: int = 3,
    use_direction: bool = True,
    R_ref: Optional[np.ndarray] = None,
    y_down_constraint: bool = True,
    collect_diag: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """GOOPPnPLで姿勢推定する（ロバスト推定なし）。

    選択規則: 全候補解（J昇順）から
      is_real（実数解）& is_front（点群がカメラ前方）
      & r12<0（カメラY軸が下向き、y_down_constraint 有効時）
      & 再投影残差が有限
    を満たす最初の候補（=J最小）を採用し、OIPnPL で反復改良する。
    反転解は「上下反転系統（r12>0）」か「平面裏側への鏡映系統（front=False）」に
    落ちるため、このフィルタで選択の土俵から排除される（pose_diag解析で実証）。

    フィルタを通る候補が無い場合は上位2解から R_ref に回転が近い方
    （参照が無ければ残差の小さい方）を選ぶ。
    R_ref: 参照回転行列（前フレーム姿勢など）。フォールバック時のみ使用。
    y_down_constraint: カメラY軸が常に下向き（ワールドZが上向き）という
    リグ前提を使うか。構図が変わる場合は config で無効化する。
    collect_diag: True なら全候補解を POSE_DIAG に蓄積する。"""
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

    all_sols = None
    try:
        if hasattr(_goppnpl, "GOOPPnPL_all"):
            R1, t1, R2, t2, all_sols = _goppnpl.GOOPPnPL_all([], [], Pl, i_Pl, use_flag)
        else:
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

    # ── 全候補からの選択（docstring の選択規則を参照）──
    sel = ("refined", 1)
    R_out = t_out = None
    if all_sols is not None:
        for i, (R_c, t_c, J_c, is_real, is_front) in enumerate(all_sols):  # J昇順
            if not (is_real and is_front):
                continue
            R_c = np.asarray(R_c, dtype=np.float64)
            t_c = np.asarray(t_c, dtype=np.float64).flatten()
            if y_down_constraint and R_c[1, 2] >= 0.0:
                continue
            if _total_residual(R_c, t_c) == float('inf'):
                continue
            sel = ("cand", i)
            if i == 0:            # idx 0/1 の改良版は R1/R2 として計算済み
                R_out, t_out = R1, t1
            elif i == 1:
                R_out, t_out = R2, t2
            elif hasattr(_goppnpl, "OIPnPL_refine"):
                try:
                    R_r, t_r = _goppnpl.OIPnPL_refine([], [], Pl, i_Pl, use_flag, R_c)
                    R_out = np.asarray(R_r, dtype=np.float64)
                    t_out = np.asarray(t_r, dtype=np.float64).flatten()
                except Exception:
                    R_out, t_out = R_c, t_c
            else:
                R_out, t_out = R_c, t_c
            break

    # フォールバック: フィルタを通る候補が無い場合は上位2解から
    # R_ref に回転が近い方（参照が無ければ残差の小さい方）を選ぶ。
    if R_out is None:
        if R_ref is not None:
            a1 = _rotation_angle_deg(R1, R_ref)
            a2 = _rotation_angle_deg(R2, R_ref)
            idx = 1 if a1 <= a2 else 2
        else:
            idx = 2 if _total_residual(R2, t2) < _total_residual(R1, t1) else 1
        sel = ("refined", idx)
        R_out, t_out = (R1, t1) if idx == 1 else (R2, t2)

    if collect_diag and all_sols is not None:
        _append_pose_diag(all_sols, R1, t1, R2, t2,
                          _total_residual(R1, t1), _total_residual(R2, t2),
                          sel, correspondences, K, R_ref)

    return R_out, t_out


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────────────────

def estimate_from_lines(
    R_init:           np.ndarray,   # 参照回転行列（前フレーム姿勢）。鏡映曖昧性の解消に使用
    correspondences:  list,
    K:                np.ndarray,
    min_lines:        int   = 3,
    use_direction:    bool  = True,
    robust_method:    str   = "magsac",  # "magsac" / "ransac" / "none"
    sigma_max_px:     float = 20.0,
    ransac_thresh_px: float = 10.0,
    y_down_constraint: bool = True,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    2D-3D 線分対応から姿勢 (R, t) を推定する。

    R_init           : 参照回転行列。候補フィルタが全滅した場合のフォールバックで
                       回転が近い方を選ぶのに使う。None 可
    correspondences  : [(ProjectedLine, KeyLine, kl_idx), ...] の対応リスト
    K                : 3×3 カメラ内部行列
    robust_method    : ロバスト推定手法 ("magsac" / "ransac" / "none")
    sigma_max_px     : MAGSACのノイズスケール上限 [px]
    ransac_thresh_px : RANSACのインライア判定閾値 [px]
    y_down_constraint: カメラY軸が常に下向きというリグ前提で候補解を
                       フィルタする（鏡映反転解対策）。構図が変わる場合は無効化
    """
    if not _GOPPNPL_AVAILABLE:
        return None
    if len(correspondences) < min_lines:
        return None

    R_ref = None
    if R_init is not None:
        R_arr = np.asarray(R_init, dtype=np.float64)
        if R_arr.shape == (3, 3):
            R_ref = R_arr

    if robust_method != "none" and len(correspondences) > min_lines:
        n_iter = _compute_magsac_iters(len(correspondences))
        if robust_method == "magsac":
            inlier_idx = _magsac_inliers(
                correspondences, K, min_lines, n_iter, sigma_max_px, use_direction,
                R_ref, y_down_constraint,
            )
        else:  # ransac
            inlier_idx = _ransac_inliers(
                correspondences, K, min_lines, n_iter, ransac_thresh_px, use_direction,
                R_ref, y_down_constraint,
            )
        if inlier_idx:
            correspondences = [correspondences[i] for i in inlier_idx]

    return _estimate_pose_core(correspondences, K, min_lines, use_direction, R_ref,
                               y_down_constraint, collect_diag=True)
