"""
test_distortion_fix_accuracy.py
================================
line_pose.py に追加した歪み補正（dist引数）が姿勢推定精度を
改善するか悪化させるかを、合成データで検証する。

手順:
  1. 実際の lines3d.csv・カメラ内部パラメータ・歪み係数を使う
  2. 既知のカメラ姿勢(R_gt, t_gt)を複数生成
  3. 3D線分を「実カメラ（歪みあり）」の順方向歪みモデルで2Dに投影し、
     EDLinesが実際に検出するであろう「歪んだ画素座標」を合成する
  4. その2D線分を dist=None（旧動作）/ dist=実係数（修正後）で
     estimate_from_lines に渡し、姿勢を推定する
  5. 真値との回転誤差・並進誤差を比較する

  ノイズなし版: 純粋に「歪み補正の有無」による幾何的バイアスだけを見る
  ノイズあり版: 実運用に近い検出ノイズ下での挙動を見る
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

import line_pose
from line_pose import estimate_from_lines, _rotation_angle_deg
from projector import load_lines3d_csv

rng = np.random.default_rng(42)

# ── 実際のカメラパラメータ ──────────────────────────────────────
K = np.array([[1694.400466, 0, 950.015140],
              [0, 1690.299550, 511.140341],
              [0, 0, 1.0]])
DIST = np.array([0.174029, -0.737755, -0.000809, 0.000720, 1.135290])  # 新キャリブ
IMG_W, IMG_H = 1920, 1080

lines3d = load_lines3d_csv(str(REPO / "data/lines3d.csv"))
print(f"lines3d: {len(lines3d)}本")


def rot_xyz(ax, ay, az):
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(ax), -s(ax)], [0, s(ax), c(ax)]])
    Ry = np.array([[c(ay), 0, s(ay)], [0, 1, 0], [-s(ay), 0, c(ay)]])
    Rz = np.array([[c(az), -s(az), 0], [s(az), c(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def forward_distort(x, y, dist):
    """正規化座標(x,y)に順方向の歪みを適用し画素座標を返す（projector.pyと同一式）。"""
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]
    return u, v


def make_synthetic_frame(R_gt, t_gt, noise_px=0.0):
    """R_gt,t_gtで3D線分を実カメラ(歪みあり)として投影し、画角内に収まる
    対応(correspondences)を合成する。"""
    corr = []
    for idx, (p1w, p2w) in enumerate(lines3d):
        pts_img = []
        ok = True
        for pw in (p1w, p2w):
            Xc = R_gt @ pw + t_gt
            if Xc[2] <= 0:
                ok = False
                break
            x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
            u, v = forward_distort(x, y, DIST)
            if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                ok = False
                break
            pts_img.append((u, v))
        if not ok:
            continue
        (u1, v1), (u2, v2) = pts_img
        if noise_px > 0:
            u1 += rng.normal(0, noise_px); v1 += rng.normal(0, noise_px)
            u2 += rng.normal(0, noise_px); v2 += rng.normal(0, noise_px)
        proj = SimpleNamespace(p1_3d=p1w, p2_3d=p2w, pt1_2d=(u1, v1), pt2_2d=(u2, v2))
        kl = SimpleNamespace(startPointX=u1, startPointY=v1, endPointX=u2, endPointY=v2)
        corr.append((proj, kl, idx))
    return corr


# 実際にマーカーから復元された基準姿勢（このconversation内の実測値）。
# これを中心に小さく姿勢を揺らして合成フレームを作る＝実際の部屋を見ている
# 保証がある（frame-to-frameの変動を模した現実的な設定）。
_R_BASE = np.array([
    [0.97821296, -0.20751992, 0.00590636],
    [-0.01548828, -0.1013208, -0.99473324],
    [0.20702539, 0.97296947, -0.10232744],
])
_T_BASE = np.array([0.87147909, 1.03579342, 4.25027073])


def rand_pose_near_marker(max_rot_deg=8.0, max_trans=0.4):
    """実測の基準姿勢を小さく揺らした姿勢を生成する（多くの線分が視野に入る）。"""
    d_ax, d_ay, d_az = np.radians(rng.uniform(-max_rot_deg, max_rot_deg, 3))
    dR = rot_xyz(d_ax, d_ay, d_az)
    R_gt = dR @ _R_BASE
    c_base = -_R_BASE.T @ _T_BASE
    c_gt = c_base + rng.uniform(-max_trans, max_trans, 3)
    t_gt = -R_gt @ c_gt
    return R_gt, t_gt


def undistort_corr(corr):
    """対応リストの検出線分端点を歪み補正する（画像側を一度だけ補正するのと
    数学的に等価。line_pose.py 自体は変更せず、テスト側で座標を補正してから渡す）。"""
    pts = []
    for item in corr:
        kl = item[1]
        pts.append([kl.startPointX, kl.startPointY])
        pts.append([kl.endPointX, kl.endPointY])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, DIST, P=K).reshape(-1, 2)
    out = []
    for i, item in enumerate(corr):
        sx, sy = und[2 * i]
        ex, ey = und[2 * i + 1]
        new_kl = SimpleNamespace(startPointX=sx, startPointY=sy, endPointX=ex, endPointY=ey)
        out.append((item[0], new_kl) + tuple(item[2:]))
    return out


def run_trials(n_trials, noise_px, label):
    errs_off, errs_on = [], []
    paired_diff = []   # off - on （正なら修正後の方が良い）
    n_win_on = n_win_off = n_tie = 0
    n_corr_list = []
    skipped = 0
    for _ in range(n_trials):
        R_gt, t_gt = rand_pose_near_marker()
        corr = make_synthetic_frame(R_gt, t_gt, noise_px=noise_px)
        if len(corr) < 4:
            skipped += 1
            continue
        n_corr_list.append(len(corr))

        res_off = estimate_from_lines(None, corr, K,
                                      min_lines=3, robust_method="none")
        res_on = estimate_from_lines(None, undistort_corr(corr), K,
                                     min_lines=3, robust_method="none")
        if res_off is None or res_on is None:
            continue
        e_off = _rotation_angle_deg(res_off[0], R_gt)
        e_on = _rotation_angle_deg(res_on[0], R_gt)
        errs_off.append(e_off)
        errs_on.append(e_on)
        paired_diff.append(e_off - e_on)
        if e_on < e_off - 1e-9:
            n_win_on += 1
        elif e_off < e_on - 1e-9:
            n_win_off += 1
        else:
            n_tie += 1

    def stat(name, xs):
        if not xs:
            print(f"  {name}: データなし"); return
        xs = np.array(xs)
        print(f"  {name}: 平均={xs.mean():.4f}°  中央値={np.median(xs):.4f}°  "
              f"最大={xs.max():.4f}°  (n={len(xs)})")

    print(f"\n=== {label} (試行{n_trials}, スキップ{skipped}, 平均対応本数{np.mean(n_corr_list):.1f}) ===")
    stat("回転誤差 [歪み補正なし=旧動作]", errs_off)
    stat("回転誤差 [歪み補正あり=修正後]", errs_on)
    if paired_diff:
        pd = np.array(paired_diff)
        print(f"  ペア差(旧-新): 平均={pd.mean():+.4f}°  (正=修正後が優位)")
        print(f"  勝敗: 修正後が良い {n_win_on} / 旧が良い {n_win_off} / 同等 {n_tie}")


print("\n" + "="*70)
print("実験1: 検出ノイズなし（純粋な幾何バイアスの検証）")
print("="*70)
run_trials(300, noise_px=0.0, label="ノイズなし")

print("\n" + "="*70)
print("実験2: 検出ノイズあり（実運用に近い条件, 0.5px）")
print("="*70)
run_trials(300, noise_px=0.5, label="ノイズ0.5px")

print("\n" + "="*70)
print("実験3: 検出ノイズあり（1.0px）")
print("="*70)
run_trials(300, noise_px=1.0, label="ノイズ1.0px")

print("\n" + "="*70)
print("実験4: 検出ノイズあり（2.0px, 厳しめの条件）")
print("="*70)
run_trials(300, noise_px=2.0, label="ノイズ2.0px")
