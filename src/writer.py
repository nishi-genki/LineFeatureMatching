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


def draw_matches_side_by_side(
    frame1,
    kl1,
    frame2,
    kl2,
    matches,
    cfg: dict,
    proj_lines1: list | None = None,
    proj_lines2: list | None = None,
    tracked_kl_indices2: set | None = None,
    stage: int = 4,
) -> np.ndarray:
    """2 フレームを横並びにして線分と対応線を描画する。
    tracked_kl_indices2: 記述子トラッキングで引き継がれた右フレームの kl インデックス集合。
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
    color_tracked = (0, 0, 255)  # 赤: 記述子トラッキングで引き継がれた線分
    color_match = (0, 165, 255)

    tracked = tracked_kl_indices2 or set()

    if only_matched:
        matched_idx1 = {m.queryIdx for m in matches[:max_draw]}
        matched_idx2 = {m.trainIdx for m in matches[:max_draw]}
        for i in matched_idx1:
            kl = kl1[i]
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color_left,
                line_thickness,
            )
        for i in matched_idx2:
            kl = kl2[i]
            color = color_tracked if i in tracked else color_right
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color,
                line_thickness,
            )
    else:
        for kl in kl1:
            cv2.line(
                canvas,
                (int(kl.startPointX), int(kl.startPointY)),
                (int(kl.endPointX), int(kl.endPointY)),
                color_left,
                line_thickness,
            )
        for i, kl in enumerate(kl2):
            color = color_tracked if i in tracked else color_right
            cv2.line(
                canvas,
                (int(kl.startPointX) + w1, int(kl.startPointY)),
                (int(kl.endPointX) + w1, int(kl.endPointY)),
                color,
                line_thickness,
            )

    if vis_cfg.get("show_match_lines", True):
        for m in matches[:max_draw]:
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
    proj_cfg = cfg.get("projection", {})
    proj_color = tuple(proj_cfg.get("color", [0, 0, 255]))
    proj_thick = proj_cfg.get("thickness", 2)

    for pl in proj_lines1 or []:
        u1, v1 = pl.pt1_2d
        u2, v2 = pl.pt2_2d
        cv2.line(canvas, (int(u1), int(v1)), (int(u2), int(v2)), proj_color, proj_thick, cv2.LINE_AA)

    for pl in proj_lines2 or []:
        u1, v1 = pl.pt1_2d
        u2, v2 = pl.pt2_2d
        cv2.line(canvas, (int(u1) + w1, int(v1)), (int(u2) + w1, int(v2)), proj_color, proj_thick, cv2.LINE_AA)

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

    # ── ステージインジケータ ────────────────────────────────────────
    stage_labels = [
        (1, "1.Detect"),
        (2, "2.LBD"),
        (3, "3.3DProj"),
        (4, "4.2D-3DProp"),
    ]
    color = (100, 220, 100)  # 緑
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = text_font_scale * 0.85
    fthick = text_thickness
    x, y_stage = 10, 50
    for s_num, s_label in stage_labels:
        active = s_num <= stage
        prefix = "[*] " if active else "[ ] "  # アクティブなステージには [*]、非アクティブには [ ] を付ける
        text = prefix + s_label
        (tw, _), _ = cv2.getTextSize(text, font, fscale, fthick)
        cv2.putText(canvas, text, (x, y_stage), font, fscale, color, fthick, cv2.LINE_AA)
        x += tw + 20
    # ────────────────────────────────────────────────────────────────

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
