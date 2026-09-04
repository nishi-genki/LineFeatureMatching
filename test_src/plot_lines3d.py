"""lines3d.csv を3D復元して複数視点で可視化する。"""
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

from projector import load_lines3d_csv

lines = load_lines3d_csv(str(REPO / "data/lines3d.csv"))
print(f"線分数: {len(lines)}")


def load_dat(path):
    pts = []
    for line in open(path):
        line = line.strip()
        if line:
            pts.append(list(map(float, line.split())))
    return np.array(pts)


marker = load_dat(str(REPO / "markers/aruco/id0_DICT_4X4_50.dat"))

fig = plt.figure(figsize=(20, 16))


def draw(ax, elev, azim, title):
    for i, (p1, p2) in enumerate(lines):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=1.5)
        mid = (p1 + p2) / 2
        ax.text(mid[0], mid[1], mid[2], str(i), fontsize=7, color='red')
    mk = np.vstack([marker, marker[0]])
    ax.plot(mk[:, 0], mk[:, 1], mk[:, 2], 'g-', linewidth=2, label='ArUco marker')
    ax.scatter(*marker[0], color='green', s=40)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left', fontsize=8)

    allpts = np.vstack([p for pair in lines for p in pair] + [marker])
    mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)


ax1 = fig.add_subplot(221, projection='3d')
draw(ax1, elev=25, azim=-60, title='俯瞰(アイソメ)')

ax2 = fig.add_subplot(222, projection='3d')
draw(ax2, elev=90, azim=-90, title='上から (平面図: X-Y)')

ax3 = fig.add_subplot(223, projection='3d')
draw(ax3, elev=0, azim=-90, title='正面 (立面図: X-Z, Y方向から見る)')

ax4 = fig.add_subplot(224, projection='3d')
draw(ax4, elev=0, azim=0, title='側面 (Y-Z, X方向から見る)')

plt.tight_layout()
out = REPO / "lines3d_reconstruction.png"
plt.savefig(out, dpi=130)
print(f"保存: {out}")

print("\n線分一覧:")
for i, (p1, p2) in enumerate(lines):
    length = np.linalg.norm(p2 - p1)
    print(f"  {i:>2}: {p1} -> {p2}  (長さ {length:.2f}m)")
