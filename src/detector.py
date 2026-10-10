"""
detector.py
===========
線分検出・向き正規化・LBD記述子計算。

対応検出器:
    - LSD (Line Segment Detector)
    - EDLines (Edge Drawing Lines)

近傍線分の統合（detection.merge）:
    歪み・遮蔽・検出器の癖で1本の実エッジが複数の短い断片に分かれて
    検出されるのを防ぐため、ほぼ同一直線上にあり隙間の小さい線分同士を
    1本に統合してから長さフィルタ・本数制限を適用する。太く長い線分が
    得られることで、フレーム間のLBD対応（本数・記述子）も安定する。
"""

import math

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 初期化
# ─────────────────────────────────────────────


def build_detectors(cfg: dict):
    """cfg の detection / descriptor をもとに検出器・記述子・マッチャーを生成する。"""
    detector_type = cfg["detection"]["detector"]

    if detector_type == "lsd":
        lsd_cfg = cfg["detection"]["lsd"]
        # LSDParam.scale は LSD の記述子スケール（config の lsd.scale）
        lsd_params = cv2.line_descriptor.LSDParam()
        lsd_params.scale = lsd_cfg["scale"]
        lsd_params.density_th = lsd_cfg["density_th"]
        detector = cv2.line_descriptor.LSDDetector_createLSDDetectorWithParams(lsd_params)
        descriptor = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
        return detector, descriptor, matcher
    elif detector_type == "edlines":
        detector = cv2.ximgproc.createEdgeDrawing()
        edlines_cfg = cfg["detection"]["edlines"]

        params = detector.Params()
        params.LineFitErrorThreshold = edlines_cfg["LineFitErrorThreshold"]
        params.MaxDistanceBetweenTwoLines = edlines_cfg["MaxDistanceBetweenTwoLines"]
        # params.MaxErrorThreshold = 2.0  # デフォルト1.3 → 延長許容誤差を緩める
        # params.EdgeDetectionOperator = cv2.ximgproc.EdgeDrawing_SCHARR  # 精度の高いオペレータ

        detector.setParams(params)
        descriptor = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()
        matcher = cv2.line_descriptor.BinaryDescriptorMatcher()
        return detector, descriptor, matcher
    else:
        raise ValueError("detector は 'lsd' または 'edlines' のみ指定可能です。")


# ─────────────────────────────────────────────
# 向き正規化
# ─────────────────────────────────────────────


def normalize_line_directions(keylines, cfg: dict):
    """線の向きを「左から右」（垂直なら「上から下」）に統一する。"""
    eps = cfg["geometry"]["vertical_epsilon"]
    for kl in keylines:
        if (kl.startPointX > kl.endPointX) or (
            abs(kl.startPointX - kl.endPointX) < eps and kl.startPointY > kl.endPointY
        ):
            kl.startPointX, kl.endPointX = kl.endPointX, kl.startPointX
            kl.startPointY, kl.endPointY = kl.endPointY, kl.startPointY
            kl.angle = kl.angle + np.pi if kl.angle < 0 else kl.angle - np.pi
    return keylines


# ─────────────────────────────────────────────
# 検出・記述
# ─────────────────────────────────────────────


def detect_lines(gray: np.ndarray, detector, cfg: dict, valid_mask: np.ndarray = None) -> list:
    """グレースケール画像から KeyLine のリストを返す（向き正規化付き）。記述子は計算しない。

    valid_mask: 歪み補正(cv2.remap)で実ソースが存在する画素が非ゼロのマスク。
    指定すると、検出線分のうち無効領域（BORDER_REPLICATE等で捏造された
    領域）にかかる部分をクリップしてから統合・長さフィルタを適用する。
    """
    detector_type = cfg["detection"]["detector"]
    max_lines = cfg["detection"]["max_lines"]

    if detector_type == "lsd":
        return _detect_lsd(gray, detector, max_lines, cfg["detection"]["lsd"], cfg, valid_mask)
    elif detector_type == "edlines":
        return _detect_edlines(gray, detector, max_lines, cfg["detection"]["edlines"]["min_length"], cfg, valid_mask)
    else:
        raise ValueError("detector は 'lsd' または 'edlines' のみ指定可能です。")


# ─────────────────────────────────────────────
# 有効領域マスクによるクリップ
# ─────────────────────────────────────────────


def _clip_segments_to_mask(
    segs: list[tuple[float, float, float, float]],
    valid_mask: np.ndarray,
    min_length: float = 3.0,
) -> list[tuple[float, float, float, float]]:
    """各線分を有効領域マスクでクリップする。

    歪み補正(cv2.remap)で BORDER_REPLICATE 等を使うと、画角の外側
    （実ソースが存在しない領域）は近傍画素の引き伸ばしで滑らかに埋まるが、
    その内容はシーンを正しく再現したものではない。繰り返し模様（ブラインド
    窓など）が引き伸ばされて実エッジの延長のように検出されることがあるため、
    線分のうち無効領域にかかる区間を切り捨て、有効な部分だけを残す。
    """
    if valid_mask is None:
        return segs
    h, w = valid_mask.shape
    out: list[tuple[float, float, float, float]] = []
    for (x1, y1, x2, y2) in segs:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-6:
            continue
        n = max(int(length), 2)
        ts = np.linspace(0.0, 1.0, n + 1)
        xs = x1 + (x2 - x1) * ts
        ys = y1 + (y2 - y1) * ts
        xi = np.clip(xs.round().astype(int), 0, w - 1)
        yi = np.clip(ys.round().astype(int), 0, h - 1)
        ok = valid_mask[yi, xi] > 0
        valid_idx = np.where(ok)[0]
        if len(valid_idx) == 0:
            continue
        i0, i1 = valid_idx[0], valid_idx[-1]
        nx1, ny1, nx2, ny2 = xs[i0], ys[i0], xs[i1], ys[i1]
        if math.hypot(nx2 - nx1, ny2 - ny1) >= min_length:
            out.append((float(nx1), float(ny1), float(nx2), float(ny2)))
    return out


# ─────────────────────────────────────────────
# 近傍線分の統合
# ─────────────────────────────────────────────


def _merge_collinear_segments(
    segs: list[tuple[float, float, float, float]],
    angle_th_deg: float,
    perp_th_px: float,
    gap_th_px: float,
) -> list[tuple[float, float, float, float]]:
    """ほぼ同一直線上（角度差 angle_th_deg 以内・垂直距離 perp_th_px 以内）にあり、
    直線方向の隙間が gap_th_px 以内の線分同士を1本に統合する。
    Union-Find で連結成分を作り、各成分は最長メンバーの方向を基準に
    全端点を投影した最小〜最大区間で1本の線分に置き換える。"""
    n = len(segs)
    if n <= 1:
        return list(segs)

    angle_th = math.radians(angle_th_deg)
    pts1 = np.array([(s[0], s[1]) for s in segs], dtype=np.float64)
    pts2 = np.array([(s[2], s[3]) for s in segs], dtype=np.float64)
    d = pts2 - pts1
    lengths = np.hypot(d[:, 0], d[:, 1])
    safe_lengths = np.where(lengths < 1e-6, 1.0, lengths)
    dirs = d / safe_lengths[:, None]
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        p1i, ui, ai = pts1[i], dirs[i], angles[i]
        ni = np.array([-ui[1], ui[0]])
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            da = abs(ai - angles[j]) % math.pi
            da = min(da, math.pi - da)
            if da > angle_th:
                continue

            p1j, p2j = pts1[j], pts2[j]
            if max(abs(np.dot(p1j - p1i, ni)), abs(np.dot(p2j - p1i, ni))) > perp_th_px:
                continue

            ti = sorted((0.0, float(lengths[i])))
            tj = sorted((float(np.dot(p1j - p1i, ui)), float(np.dot(p2j - p1i, ui))))
            gap = max(tj[0] - ti[1], ti[0] - tj[1], 0.0)
            if gap > gap_th_px:
                continue

            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[tuple[float, float, float, float]] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(segs[idxs[0]])
            continue
        ref = max(idxs, key=lambda i: lengths[i])
        p1r, ur = pts1[ref], dirs[ref]
        proj: list[float] = []
        for i in idxs:
            proj.append(float(np.dot(pts1[i] - p1r, ur)))
            proj.append(float(np.dot(pts2[i] - p1r, ur)))
        tmin, tmax = min(proj), max(proj)
        new_p1 = p1r + tmin * ur
        new_p2 = p1r + tmax * ur
        merged.append((float(new_p1[0]), float(new_p1[1]),
                       float(new_p2[0]), float(new_p2[1])))
    return merged


def _apply_merge(segs: list[tuple[float, float, float, float]], cfg: dict) -> list[tuple[float, float, float, float]]:
    """cfg["detection"]["merge"] が有効なら統合を適用する。"""
    merge_cfg = cfg["detection"].get("merge", {})
    if not merge_cfg.get("enable", False):
        return segs
    return _merge_collinear_segments(
        segs,
        angle_th_deg=float(merge_cfg.get("angle_th_deg", 3.0)),
        perp_th_px=float(merge_cfg.get("perp_dist_th_px", 4.0)),
        gap_th_px=float(merge_cfg.get("gap_th_px", 20.0)),
    )


def _build_keylines(segs: list[tuple[float, float, float, float]]) -> list:
    """(x1,y1,x2,y2) のリストから KeyLine のリストを作る（response=長さ）。"""
    keylines = []
    for idx, (x1, y1, x2, y2) in enumerate(segs):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        kl = cv2.line_descriptor.KeyLine()
        kl.startPointX = x1
        kl.startPointY = y1
        kl.endPointX = x2
        kl.endPointY = y2
        kl.sPointInOctaveX = x1
        kl.sPointInOctaveY = y1
        kl.ePointInOctaveX = x2
        kl.ePointInOctaveY = y2
        kl.angle = np.arctan2(dy, dx)
        kl.lineLength = length
        kl.numOfPixels = int(length)
        kl.class_id = idx
        kl.octave = 0
        kl.response = length
        kl.pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        kl.size = length
        keylines.append(kl)
    return keylines


def describe_lines(gray: np.ndarray, keylines: list, descriptor):
    """KeyLine リストから LBD 記述子を計算して返す。keylines が空なら ([], None)。"""
    if not keylines:
        return [], None
    try:
        keylines, descs = descriptor.compute(gray, keylines)
    except cv2.error as e:
        print(f"[WARN] descriptor.compute failed: {e}")
        return [], None
    if descs is None or len(descs) == 0:
        return [], None
    return keylines, descs


def _detect_lsd(gray, detector, max_lines, lsd_cfg: dict, cfg: dict, valid_mask=None) -> list:
    keylines = detector.detect(
        gray,
        scale=lsd_cfg["octave_scale"],
        numOctaves=lsd_cfg["num_octaves"],
        mask=None,
    )
    if not keylines:
        return []

    segs = [(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY) for kl in keylines]
    # 統合の前に無効領域（歪み補正で捏造された範囲）でクリップする。
    # 統合は端点が近いものを繋げるため、クリップ前にやると捏造部分を
    # 挟んで誤統合しうる。
    segs = _clip_segments_to_mask(segs, valid_mask)
    segs = _apply_merge(segs, cfg)

    # 統合後は長さで再構成（統合前の NFA response は統合により意味を失うため）。
    segs.sort(key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)
    keylines = _build_keylines(segs[:max_lines])
    if not keylines:
        return []
    return normalize_line_directions(keylines, cfg)


def _detect_edlines(gray, detector, max_lines, min_length, cfg: dict, valid_mask=None) -> list:
    detector.detectEdges(gray)
    lines = detector.detectLines()

    if lines is None or len(lines) == 0:
        return []

    segs = []
    for line in lines:
        if isinstance(line[0], (np.ndarray, list)):
            x1, y1, x2, y2 = map(float, line[0])
        else:
            x1, y1, x2, y2 = map(float, line)
        segs.append((x1, y1, x2, y2))

    # 統合の前に無効領域（歪み補正で捏造された範囲）でクリップする。
    # 統合は端点が近いものを繋げるため、クリップ前にやると捏造部分を
    # 挟んで誤統合しうる。
    segs = _clip_segments_to_mask(segs, valid_mask)

    # 長さフィルタ・本数制限の前に統合する（統合前は min_length 未満の断片
    # でも、統合後に閾値を超えれば拾えるようにするため）。
    segs = _apply_merge(segs, cfg)

    candidates = [s for s in segs if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_length]
    candidates.sort(key=lambda s: math.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)

    keylines = _build_keylines(candidates[:max_lines])
    if not keylines:
        return []
    return normalize_line_directions(keylines, cfg)
