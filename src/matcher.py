"""
matcher.py
==========
LBD記述子間のマッチング。

アルゴリズム: BinaryDescriptorMatcher + Lowe's ratio test
"""

import math


# ─────────────────────────────────────────────
# マッチング
# ─────────────────────────────────────────────


def match_lines(descs1, descs2, matcher, ratio_thresh: float):
    """Lowe's ratio test によるマッチング。"""
    if descs1 is None or descs2 is None:
        return []
    if len(descs1) < 2 or len(descs2) < 2:
        return []

    best: dict[int, object] = {}
    for pair in matcher.knnMatch(descs1, descs2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                    best[m.trainIdx] = m
    return list(best.values())


# ─────────────────────────────────────────────
# 2D-2D 幾何整合チェック
# ─────────────────────────────────────────────


def is_match_consistent(
    kl1,
    kl2,
    angle_th_deg: float,
    dist_th_px: float,
    length_ratio_th: float,
) -> bool:
    """フレーム間でマッチした検出線分同士の幾何学的整合性を判定する。
    角度差・中点距離・長さ比がすべて閾値以内なら True。
    姿勢や3D情報に依存しない 2D-2D 比較。"""
    # 角度差（無向: [0, π/2]）
    da = abs(kl1.angle - kl2.angle) % math.pi
    da = min(da, math.pi - da)
    if da > math.radians(angle_th_deg):
        return False

    # 中点間距離
    mx1 = (kl1.startPointX + kl1.endPointX) / 2.0
    my1 = (kl1.startPointY + kl1.endPointY) / 2.0
    mx2 = (kl2.startPointX + kl2.endPointX) / 2.0
    my2 = (kl2.startPointY + kl2.endPointY) / 2.0
    if math.hypot(mx1 - mx2, my1 - my2) > dist_th_px:
        return False

    # 長さ比
    l1, l2 = float(kl1.lineLength), float(kl2.lineLength)
    if min(l1, l2) < 1e-6:
        return False
    if max(l1, l2) / min(l1, l2) > length_ratio_th:
        return False

    return True
