"""
detector.py
===========
線分検出・向き正規化・LBD記述子計算。

対応検出器:
    - LSD (Line Segment Detector)
    - EDLines (Edge Drawing Lines)
"""

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


def detect_lines(gray: np.ndarray, detector, cfg: dict) -> list:
    """グレースケール画像から KeyLine のリストを返す（向き正規化付き）。記述子は計算しない。"""
    detector_type = cfg["detection"]["detector"]
    max_lines = cfg["detection"]["max_lines"]

    if detector_type == "lsd":
        return _detect_lsd(gray, detector, max_lines, cfg["detection"]["lsd"], cfg)
    elif detector_type == "edlines":
        return _detect_edlines(gray, detector, max_lines, cfg["detection"]["edlines"]["min_length"], cfg)
    else:
        raise ValueError("detector は 'lsd' または 'edlines' のみ指定可能です。")


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


def _detect_lsd(gray, detector, max_lines, lsd_cfg: dict, cfg: dict) -> list:
    keylines = detector.detect(
        gray,
        scale=lsd_cfg["octave_scale"],
        numOctaves=lsd_cfg["num_octaves"],
        mask=None,
    )
    if not keylines:
        return []

    for kl in keylines:
        if not hasattr(kl, "octave") or kl.octave is None:
            kl.octave = 0

    keylines = sorted(keylines, key=lambda k: k.response, reverse=True)[:max_lines]
    return normalize_line_directions(keylines, cfg)


def _detect_edlines(gray, detector, max_lines, min_length, cfg: dict) -> list:
    detector.detectEdges(gray)
    lines = detector.detectLines()

    if lines is None or len(lines) == 0:
        return []

    candidates = []
    for line in lines:
        if isinstance(line[0], (np.ndarray, list)):
            x1, y1, x2, y2 = map(float, line[0])
        else:
            x1, y1, x2, y2 = map(float, line)
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            candidates.append((length, (x1, y1, x2, y2)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_lines = [c[1] for c in candidates[:max_lines]]

    keylines = []
    for idx, (x1, y1, x2, y2) in enumerate(selected_lines):
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

    if not keylines:
        return []

    return normalize_line_directions(keylines, cfg)
