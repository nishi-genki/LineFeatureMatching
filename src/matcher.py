"""
matcher.py
==========
LBD記述子間のマッチング。

アルゴリズム: BinaryDescriptorMatcher + Lowe's ratio test
"""


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
