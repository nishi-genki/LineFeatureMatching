"""
writer.py
=========
可視化フレームの生成と対応情報のCSV書き込み。
"""

import csv

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────


def _draw_stage_indicator(canvas, stage: int, fscale: float, fthick: int):
    """左上にステージ進捗（[*] 1.Detect [*] 2.LBD ...）を描画する。"""
    stage_labels = [
        (1, "1.Detect"),
        (2, "2.LBD"),
        (3, "3.3DProj"),
        (4, "4.2D-3DProp"),
    ]
    color = (100, 220, 100)  # 緑
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y_stage = 10, 50
    for s_num, s_label in stage_labels:
        active = s_num <= stage
        prefix = "[*] " if active else "[ ] "
        text = prefix + s_label
        (tw, _), _ = cv2.getTextSize(text, font, fscale, fthick)
        cv2.putText(canvas, text, (x, y_stage), font, fscale, color, fthick, cv2.LINE_AA)
        x += tw + 20


def draw_world_axes(canvas, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                     length: float = 1.0, thickness: int = 3, x_offset: int = 0):
    """世界座標系の原点と X/Y/Z 軸を姿勢 (R, t) で画像に投影して描画する
    （先行研究と同様、精度の目視確認用）。色は X=赤, Y=緑, Z=青（cv2.drawFrameAxes と同じ配色）。
    軸先端がカメラ後方（Z<=0）の場合はその軸の描画を省略する。"""
    pts_3d = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
        [0.0, 0.0, length],
    ], dtype=np.float64)

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).flatten()
    cam_pts = (R @ pts_3d.T).T + t
    if cam_pts[0, 2] <= 0:
        return

    uv = (K @ cam_pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    origin = (int(uv[0, 0]) + x_offset, int(uv[0, 1]))

    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=赤, Y=緑, Z=青 (BGR)
    for i, color in enumerate(axis_colors, start=1):
        if cam_pts[i, 2] <= 0:
            continue
        tip = (int(uv[i, 0]) + x_offset, int(uv[i, 1]))
        cv2.line(canvas, origin, tip, color, thickness, cv2.LINE_AA)


def _draw_projected_lines(canvas, proj_lines, cfg: dict, x_offset: int = 0, color=None):
    """投影3D線分を描画する。color を省略した場合は projection.color（既定）を使う。"""
    proj_cfg = cfg.get("projection", {})
    proj_color = tuple(color) if color is not None else tuple(proj_cfg.get("color", [0, 0, 255]))
    proj_thick = proj_cfg.get("thickness", 2)
    for pl in proj_lines or []:
        u1, v1 = pl.pt1_2d
        u2, v2 = pl.pt2_2d
        cv2.line(canvas, (int(u1) + x_offset, int(v1)), (int(u2) + x_offset, int(v2)),
                proj_color, proj_thick, cv2.LINE_AA)


def draw_current_frame(
    frame: np.ndarray,
    kl: list,
    cfg: dict,
    proj_lines: list | None = None,
    tracked_kl_indices: set | None = None,
    geom_kl_indices: set | None = None,
    stage: int = 4,
) -> np.ndarray:
    """現在フレームのみを描画する（検出線分・投影線分・ステージ表示、前フレームは含まない）。
    tracked_kl_indices: LBD トラッキングで 2D-3D 対応が引き継がれた kl インデックス集合。
    geom_kl_indices: 幾何学的対応で新たに 2D-3D 対応が得られた kl インデックス集合。"""
    vis_cfg = cfg["visualization"]
    line_thickness = vis_cfg["line_thickness"]
    text_font_scale = vis_cfg["text_font_scale"]
    text_thickness = vis_cfg["text_thickness"]

    canvas = frame.copy()
    color_default = (255, 0, 0)
    color_tracked = (0, 0, 255)    # 赤: LBD で 2D-3D 対応が引き継がれた線分
    color_geom    = (0, 255, 255)  # 黄: 幾何学的対応で新たに 2D-3D 対応が得られた線分

    tracked = tracked_kl_indices or set()
    geom    = geom_kl_indices    or set()

    for i, k in enumerate(kl):
        color = color_tracked if i in tracked else (color_geom if i in geom else color_default)
        cv2.line(
            canvas,
            (int(k.startPointX), int(k.startPointY)),
            (int(k.endPointX), int(k.endPointY)),
            color,
            line_thickness,
        )

    _draw_projected_lines(canvas, proj_lines, cfg)

    info = f"Frame | Lines: {len(kl)} | Tracked: {len(tracked)}"
    cv2.putText(canvas, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                text_font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    _draw_stage_indicator(canvas, stage, text_font_scale * 0.85, text_thickness)
    return canvas


def draw_projection_only(
    frame: np.ndarray,
    proj_lines: list | None,
    cfg: dict,
) -> np.ndarray:
    """現在フレームに投影3D線分だけを描画する（検出線分・対応線・テキスト注釈なし）。
    色は visualization.projection_only_color（既定: 赤）。他モードの projection.color とは独立。"""
    canvas = frame.copy()
    color = cfg.get("visualization", {}).get("projection_only_color", [0, 0, 255])
    _draw_projected_lines(canvas, proj_lines, cfg, color=color)
    return canvas


def draw_correspondence_only(
    frame: np.ndarray,
    kl: list,
    cfg: dict,
    tracked_kl_indices: set | None = None,
    geom_kl_indices: set | None = None,
    outlier_kl_indices: set | None = None,
    line3d_labels: dict | None = None,
) -> np.ndarray:
    """現在フレームに、姿勢推定（2D-3D対応）に使われた検出線分だけを描画する
    （対応を持たない検出線分・投影線分・テキスト注釈なし）。
    outlier_kl_indices を渡すと、RANSAC/MAGSAC が最終的に外れ値として除外した
    線分を visualization.correspondence_outlier_color（既定: 青）で区別して描画する
    （それ以外＝実際に姿勢推定に使われた線分は correspondence_only_color、既定: 赤）。
    tracked_kl_indices/geom_kl_indices を両方 None にすると全て同色で描画する。
    line3d_labels（kl_idx -> lines3d.csv の行番号）を渡すと、各線分の中点に
    対応する3D線分の番号を描画する（groundtruth_lines と同じ見た目）。"""
    canvas = frame.copy()
    vis_cfg = cfg.get("visualization", {})
    line_thickness = vis_cfg.get("line_thickness", 2)
    font_scale = vis_cfg.get("text_font_scale", 0.6)
    color = tuple(vis_cfg.get("correspondence_only_color", [0, 0, 255]))
    outlier_color = tuple(vis_cfg.get("correspondence_outlier_color", [255, 0, 0]))

    tracked = tracked_kl_indices or set()
    geom    = geom_kl_indices    or set()
    outlier = outlier_kl_indices or set()
    labels  = line3d_labels      or {}
    used = tracked | geom

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in used:
        k = kl[i]
        line_color = outlier_color if i in outlier else color
        sx, sy, ex, ey = k.startPointX, k.startPointY, k.endPointX, k.endPointY
        cv2.line(
            canvas,
            (int(sx), int(sy)),
            (int(ex), int(ey)),
            line_color,
            line_thickness,
        )
        if i in labels:
            label = str(labels[i])
            mx, my = int((sx + ex) / 2), int((sy + ey) / 2)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(canvas, (mx - 1, my - th - 3), (mx + tw + 1, my + 2), (0, 0, 0), -1)
            cv2.putText(canvas, label, (mx, my), font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_matches_side_by_side(
    frame1,
    kl1,
    frame2,
    kl2,
    matches,
    cfg: dict,
    proj_lines1: list | None = None,
    proj_lines2: list | None = None,
    tracked_kl_indices1: set | None = None,
    tracked_kl_indices2: set | None = None,
    geom_kl_indices2: set | None = None,
    stage: int = 4,
) -> np.ndarray:
    """2 フレームを横並びにして線分と対応線を描画する。
    tracked_kl_indices1: LBD トラッキングで 2D-3D 対応を持つ左フレームの kl インデックス集合。
    tracked_kl_indices2: LBD トラッキングで 2D-3D 対応が引き継がれた右フレームの kl インデックス集合。
    geom_kl_indices2: 幾何学的対応で新たに 2D-3D 対応が得られた右フレームの kl インデックス集合。
    stage: 実行段階 (1〜4)。キャンバス上部にステージ表示を描く。"""
    vis_cfg = cfg["visualization"]
    max_draw = vis_cfg["max_draw"]
    only_matched = vis_cfg["only_matched"]
    line_thickness = vis_cfg["line_thickness"]
    match_thickness = vis_cfg["match_thickness"]
    text_font_scale = vis_cfg["text_font_scale"]
    text_thickness = vis_cfg["text_thickness"]

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:] = frame2

    color_left = (0, 255, 0)
    color_right = (255, 0, 0)
    color_tracked = (0, 0, 255)    # 赤: LBD で 2D-3D 対応が引き継がれた線分
    color_geom    = (0, 255, 255)  # 黄: 幾何学的対応で新たに 2D-3D 対応が得られた線分
    color_match = (0, 165, 255)

    tracked1 = tracked_kl_indices1 or set()
    tracked = tracked_kl_indices2 or set()
    geom    = geom_kl_indices2    or set()

    if only_matched:
        matched_idx1 = {m.queryIdx for m in matches[:max_draw]}
        matched_idx2 = {m.trainIdx for m in matches[:max_draw]}
        for i in matched_idx1:
            kl = kl1[i]
            color = color_tracked if i in tracked1 else color_left
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color,
                line_thickness,
            )
        for i in matched_idx2:
            kl = kl2[i]
            color = color_tracked if i in tracked else (color_geom if i in geom else color_right)
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color,
                line_thickness,
            )
    else:
        for i, kl in enumerate(kl1):
            color = color_tracked if i in tracked1 else color_left
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color,
                line_thickness,
            )
        for i, kl in enumerate(kl2):
            color = color_tracked if i in tracked else (color_geom if i in geom else color_right)
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color,
                line_thickness,
            )

    if vis_cfg.get("show_match_lines", True):
        for m in matches[:max_draw]:
            if m.trainIdx not in tracked:
                continue
            kl_a, kl_b = kl1[m.queryIdx], kl2[m.trainIdx]
            mid_a = (
                (int(kl_a.startPointX) + int(kl_a.endPointX)) // 2,
                (int(kl_a.startPointY) + int(kl_a.endPointY)) // 2,
            )
            mid_b = (
                (int(kl_b.startPointX) + int(kl_b.endPointX)) // 2 + w1,
                (int(kl_b.startPointY) + int(kl_b.endPointY)) // 2,
            )
            cv2.line(canvas, mid_a, mid_b, color_match, match_thickness, cv2.LINE_AA)

    # 投影3D線分を描画（両フレームそれぞれに）
    _draw_projected_lines(canvas, proj_lines1, cfg)
    _draw_projected_lines(canvas, proj_lines2, cfg, x_offset=w1)

    info = f"Frame pair | Lines: {len(kl1)} / {len(kl2)} | " f"Matches: {len(matches)} | Tracked: {len(tracked)}"
    cv2.putText(
        canvas,
        info,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    _draw_stage_indicator(canvas, stage, text_font_scale * 0.85, text_thickness)

    return canvas


# ─────────────────────────────────────────────
# CSV ライター
# ─────────────────────────────────────────────


class MatchCSVWriter:
    HEADER = [
        "frame_idx",
        "match_rank",
        "distance",
        "kl1_sx",
        "kl1_sy",
        "kl1_ex",
        "kl1_ey",
        "kl1_angle",
        "kl1_length",
        "kl2_sx",
        "kl2_sy",
        "kl2_ex",
        "kl2_ey",
        "kl2_angle",
        "kl2_length",
    ]

    def __init__(self, path: str):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def write(self, frame_idx: int, kl1, kl2, matches):
        for rank, m in enumerate(matches):
            a, b = kl1[m.queryIdx], kl2[m.trainIdx]
            self._w.writerow(
                [
                    frame_idx,
                    rank,
                    m.distance,
                    f"{a.startPointX:.2f}",
                    f"{a.startPointY:.2f}",
                    f"{a.endPointX:.2f}",
                    f"{a.endPointY:.2f}",
                    f"{a.angle:.4f}",
                    f"{a.lineLength:.2f}",
                    f"{b.startPointX:.2f}",
                    f"{b.startPointY:.2f}",
                    f"{b.endPointX:.2f}",
                    f"{b.endPointY:.2f}",
                    f"{b.angle:.4f}",
                    f"{b.lineLength:.2f}",
                ]
            )

    def close(self):
        self._f.close()
