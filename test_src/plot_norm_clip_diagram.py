"""
正規化座標平面上での「画角外クリップ→歪み補正」の判断基準を1枚の図にする。
projector.py の _compute_norm_bounds / _clip_segment_box が実際に使う値
（K, dist, data/lines3d.csv の index9）をそのまま使用する。
"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

# ── 実データ読み込み ─────────────────────────────────────────
cfg = json.load(open(REPO / "config/config.json"))
cam = cfg["camera"]
K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
dist = np.array(cam["dist"])
w, h = cam["width"], cam["height"]

# 画角の境界（画像四隅をundistortPointsで正規化座標に変換）
corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
corners_n = cv2.undistortPoints(corners_px, K, dist).reshape(-1, 2)
xmin, xmax = corners_n[:, 0].min(), corners_n[:, 0].max()
ymin, ymax = corners_n[:, 1].min(), corners_n[:, 1].max()
cx_n, cy_n = (xmin + xmax) / 2, (ymin + ymax) / 2
hx, hy = (xmax - xmin) / 2, (ymax - ymin) / 2
margin = 1.3
mxmin, mxmax = cx_n - hx * margin, cx_n + hx * margin
mymin, mymax = cy_n - hy * margin, cy_n + hy * margin

# index9 (壁の床際の線分) の実測正規化座標（このconversationでの実測値）
near = np.array([0.2049, 0.2440])   # r2=0.10
far  = np.array([-1.6580, 0.3762])  # r2=2.89


def clip_segment_box(x1, y1, x2, y2, xmn, xmx, ymn, ymx):
    """Liang-Barsky法（projector.pyと同一ロジック）。"""
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - xmn, xmx - x1, y1 - ymn, ymx - y1)
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)


clip_near, clip_far = clip_segment_box(*near, *far, mxmin, mxmax, mymin, mymax)
print(f"クリップ点: {clip_far}")

# ── 配色 ─────────────────────────────────────────────────
INK    = "#12192B"
GRID   = "#8C99AF"
SAFE   = "#147D64"
DANGER = "#C1531F"
ACCENT = "#3450C4"
PAPER  = "#EEF1F5"

fig, ax = plt.subplots(figsize=(10, 3.75), facecolor=PAPER)
ax.set_facecolor(PAPER)

# 軸
ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
ax.axvline(0, color=GRID, linewidth=1.2, zorder=1)
ax.grid(True, color="#D7DDE7", linewidth=0.8, zorder=0)

# 画角（四隅から算出）
fov_rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor="#5B6B85", linewidth=1.8, zorder=3)
ax.add_patch(fov_rect)
ax.text(xmax - 0.02, ymin + 0.04, "画角",
        fontsize=10.5, color="#5B6B85", va="top", ha="right")

# 有効範囲（1.3倍マージン）
margin_rect = plt.Rectangle((mxmin, mymin), mxmax - mxmin, mymax - mymin,
                            fill=False, edgecolor=ACCENT, linewidth=1.8,
                            linestyle=(0, (6, 4)), zorder=3)
ax.add_patch(margin_rect)
ax.text(mxmax - 0.02, mymin - 0.025, "有効範囲（画角×1.3）",
        fontsize=10.5, color=ACCENT, ha="right", va="bottom")

# 画角四隅の点
ax.scatter(corners_n[:, 0], corners_n[:, 1], s=22, color="#5B6B85", zorder=4)

# 元の線分全体（歪み補正前）
ax.plot([near[0], far[0]], [near[1], far[1]], color=GRID, linewidth=1.3,
       linestyle=(0, (2, 4)), zorder=2)

# 有効区間（歪み補正して使用）
ax.plot([near[0], clip_far[0]], [near[1], clip_far[1]], color=SAFE,
       linewidth=3.2, solid_capstyle="round", zorder=5)

# クリップ以遠（破棄）＝ここには歪み補正をかけない
ax.plot([clip_far[0], far[0]], [clip_far[1], far[1]], color=DANGER,
       linewidth=2.2, linestyle=(0, (1, 5)), solid_capstyle="round", zorder=5)

mid_discard = ((clip_far[0] + far[0]) / 2, (clip_far[1] + far[1]) / 2)
ax.annotate("歪み補正をかけない", mid_discard,
           xytext=(mid_discard[0] - 0.02, mid_discard[1] - 0.10),
           fontsize=12, fontweight="bold", color=DANGER, ha="center",
           arrowprops=dict(arrowstyle="-", color=DANGER, linewidth=1.2,
                          shrinkA=0, shrinkB=4))

# 端点マーカー
ax.scatter(*near, s=60, color=SAFE, zorder=6)
ax.annotate("近端  r²=0.10", near, textcoords="offset points", xytext=(8, 10),
           fontsize=11, color=INK)

ax.scatter(*clip_far, s=60, color=ACCENT, zorder=6)
ax.annotate("クリップ点\n（ここで打ち切り）", clip_far, textcoords="offset points",
           xytext=(-95, -32), fontsize=11, color=ACCENT)

ax.scatter(*far, s=60, color=DANGER, zorder=6)
ax.annotate("遠端 r²=2.89\n（22.7倍に発散）", far, textcoords="offset points",
           xytext=(-10, -34), fontsize=11, color=DANGER)

# 軸ラベル・体裁
ax.set_xlabel("x（正規化座標）", fontsize=11, color=INK)
ax.set_ylabel("y（正規化座標）", fontsize=11, color=INK)
ax.set_xlim(-1.85, 0.85)
ax.set_ylim(0.55, -0.55)   # yは下向きが正（カメラ座標系の慣例に合わせ上下反転）
ax.set_aspect("equal", adjustable="box")   # x:y を1:1に（正規化座標平面なので必須）
ax.set_title("正規化座標平面 ── index9 の遠端でクリップされる様子", fontsize=13,
             fontweight="bold", color=INK, pad=14)
for spine in ax.spines.values():
    spine.set_color("#C7D0DE")
ax.tick_params(colors=INK, labelsize=9.5)

# 凡例
legend_elems = [
    plt.Line2D([0], [0], color=SAFE, linewidth=3.2, label="歪み補正をかける区間"),
    plt.Line2D([0], [0], color=DANGER, linewidth=2.2, linestyle=(0, (1, 5)),
              label="歪み補正をかけない区間（多項式を評価しない）"),
    plt.Line2D([0], [0], marker="o", color=ACCENT, linestyle="", markersize=7,
              label="クリップ点（有効範囲の境界との交点）"),
    plt.Line2D([0], [0], color=GRID, linewidth=1.3, linestyle=(0, (2, 4)),
              label="元の3D線分（歪み補正前・全長）"),
]
leg = ax.legend(handles=legend_elems, loc="upper left", fontsize=8.8,
                bbox_to_anchor=(0.005, 0.985), ncol=1, frameon=True,
                facecolor=PAPER, edgecolor="#C7D0DE", framealpha=0.95,
                borderpad=0.7, labelspacing=0.6)

fig.text(0.5, 0.012,
        "K = [fx 1694.4, fy 1690.3, cx 950.0, cy 511.1]   "
        "dist = [0.174, -0.738, -0.001, 0.001, 1.135]   "
        "line: data/lines3d.csv index9, frame0",
        ha="center", fontsize=8.5, color="#6B7690", family="monospace")

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = REPO / "distortion_clip_diagram.png"
plt.savefig(out, dpi=180, facecolor=PAPER)
print(f"保存: {out}")
