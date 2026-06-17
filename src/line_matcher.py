"""
line_matcher.py
===============
投影3D線分と検出2D線分の幾何学的対応付け。
C++ LineSegmentMatcher の Python移植。

マッチング基準 (2パス):
    Pass 1: 角度差 < angle_th_deg、端点-直線距離の和 < dist_th
    Pass 2: しきい値を 0.8 倍に厳格化
    最終選択: オーバーラップ率が最大のもの
"""

import numpy as np
import cv2


def _angle_diff(v1: np.ndarray, v2: np.ndarray) -> float:
    """2ベクトル間の角度差 [0, π/2] を返す（方向の符号は無視）。"""
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return np.pi
    cos_a = abs(float(np.dot(v1, v2)) / (n1 * n2))
    return float(np.arccos(min(1.0, cos_a)))


def _endpoint_dist(d1: np.ndarray, d2: np.ndarray,
                   p1: np.ndarray, p2: np.ndarray) -> float:
    """det線分の両端点から proj 直線への距離の和を返す。"""
    pv = p2 - p1
    len_sq = float(np.dot(pv, pv))
    if len_sq < 1e-9:
        return float(np.linalg.norm(d1 - p1) + np.linalg.norm(d2 - p1))

    def dist(pt):
        t = float(np.dot(pt - p1, pv)) / len_sq
        proj = p1 + np.clip(t, 0.0, 1.0) * pv
        return float(np.linalg.norm(pt - proj))

    return dist(d1) + dist(d2)


def _overlap_ratio(p1: np.ndarray, p2: np.ndarray,
                   d1: np.ndarray, d2: np.ndarray) -> float:
    """det 線分が proj 線分上に投影したときのオーバーラップ率。"""
    pv = p2 - p1
    len_p = float(np.linalg.norm(pv))
    if len_p < 1e-9:
        return 0.0
    pv_n = pv / len_p
    t1 = float(np.dot(d1 - p1, pv_n)) / len_p
    t2 = float(np.dot(d2 - p1, pv_n)) / len_p
    t_min, t_max = min(t1, t2), max(t1, t2)
    return max(0.0, min(1.0, t_max) - max(0.0, t_min))


class LineMatcher2D3D:
    """
    投影3D線分と検出2D KeyLine の幾何学的対応付け。
    C++ LineSegmentMatcher と同一のマッチング基準を使用する。
    """

    def __init__(
        self,
        angle_th_deg: float = 10.0,
        dist_th: float = 60.0,
        overlap_th: float = 0.2,
    ):
        self._angle_th   = float(np.radians(angle_th_deg))
        self._dist_th    = float(dist_th)
        self._overlap_th = float(overlap_th)

    def match(self, projected_lines, keylines) -> list:
        """
        projected_lines: projector.project_lines() が返す list[ProjectedLine]
        keylines:        detect_and_describe() が返す list[KeyLine]
        戻り値: [(ProjectedLine, KeyLine), ...] の対応リスト
        """
        if not projected_lines or not keylines:
            return []

        used: set[int] = set()
        result = []

        for proj in projected_lines:
            p1 = np.array(proj.pt1_2d, dtype=np.float64)
            p2 = np.array(proj.pt2_2d, dtype=np.float64)
            pv = p2 - p1
            if np.linalg.norm(pv) < 1e-3:
                continue

            # ── Pass 1: 緩い条件で候補を収集 ──────────────────────────
            candidates = []  # (idx, angle_diff, dist, overlap)
            for i, kl in enumerate(keylines):
                if i in used:
                    continue
                d1 = np.array([kl.startPointX, kl.startPointY], dtype=np.float64)
                d2 = np.array([kl.endPointX,   kl.endPointY],   dtype=np.float64)
                dv = d2 - d1
                if np.linalg.norm(dv) < 1e-3:
                    continue

                ang = _angle_diff(pv, dv)
                if ang > self._angle_th:
                    continue

                dist = _endpoint_dist(d1, d2, p1, p2)
                if dist > self._dist_th:
                    continue

                overlap = _overlap_ratio(p1, p2, d1, d2)
                candidates.append((i, ang, dist, overlap))

            if not candidates:
                continue

            # ── Pass 2: 0.8× の厳格条件でさらに絞り込み ──────────────
            strict_ang  = 0.8 * self._angle_th
            strict_dist = 0.8 * self._dist_th
            strict = [(i, a, d, ov) for i, a, d, ov in candidates
                      if a <= strict_ang and d <= strict_dist]
            final = strict if strict else candidates

            # ── 最終選択: オーバーラップ率最大 ────────────────────────
            best = max(final, key=lambda x: x[3])
            if best[3] < self._overlap_th:
                continue

            used.add(best[0])
            result.append((proj, keylines[best[0]]))

        return result
