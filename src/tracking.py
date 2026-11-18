"""
tracking.py
============
2D-3D対応追跡・姿勢推定パイプライン本体。

- PipelineContext: 初期化時に1度だけ作られ、以後は読み取り専用で使う部品一式
  （projector / pose_estimator / line_matcher / confusable_pairs 等）。
- TrackingState: フレーム間で引き継ぐ状態（毎フレーム更新される）。
- process_frame: 1フレーム分の対応付け・姿勢推定処理本体。

pipeline.py はこのモジュールが返す結果を使って可視化・ログ記録を行うだけの
薄いオーケストレーターになる。
"""

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from matcher import is_match_consistent
from projector import LineProjector, ProjectedLine, load_lines3d_csv
from marker_pose import MarkerPoseEstimator
from line_matcher import LineMatcher2D3D
from line_pose import estimate_from_lines, _rotation_angle_deg, _line_residual


# ─────────────────────────────────────────────
# 初期化時に作られる部品一式（読み取り専用）
# ─────────────────────────────────────────────


@dataclass
class PipelineContext:
    projector: Optional[LineProjector]
    pose_estimator: Optional[MarkerPoseEstimator]
    line_matcher: Optional[LineMatcher2D3D]
    lines3d: list
    confusable_pairs: list[tuple[int, int]]
    get_pose: Callable[[int], tuple | None]
    K_mat: Optional[np.ndarray]
    stage: int
    min_line_corr: int
    marker_init_frames: int
    max_rot_jump_deg: float
    max_trans_jump: float
    lm_cfg: dict
    prop_enable: bool
    prop_angle: float
    prop_dist: float
    prop_ratio: float

    def lines3d_idx(self, p1, p2) -> int:
        """3D端点から lines3d.csv 上の行番号（0始まり）を逆引きする（ログ用）。"""
        for i, (q1, q2) in enumerate(self.lines3d):
            if np.allclose(q1, p1) and np.allclose(q2, p2):
                return i
        return -1


def _compute_confusable_pairs(
    lines3d, dist_th: float = 0.15, parallel_cos_th: float = 0.99
) -> list[tuple[int, int]]:
    """数cm程度しか離れていない平行な3D線分ペアを検出する。
    検出ノイズの大きさに対して世界座標上の差が小さすぎるペアは、
    残差ベースの対応付けでは安定して区別できない（入れ替わりが起きる）。"""
    pairs: list[tuple[int, int]] = []
    n = len(lines3d)
    for a in range(n):
        p1a, p2a = lines3d[a]
        da = p2a - p1a
        na = np.linalg.norm(da)
        if na < 1e-9:
            continue
        da = da / na
        for b in range(a + 1, n):
            p1b, p2b = lines3d[b]
            db = p2b - p1b
            nb = np.linalg.norm(db)
            if nb < 1e-9:
                continue
            db = db / nb
            if abs(float(np.dot(da, db))) < parallel_cos_th:
                continue
            mid_a, mid_b = (p1a + p2a) / 2.0, (p1b + p2b) / 2.0
            d = min(
                np.linalg.norm(p1a - p1b), np.linalg.norm(p1a - p2b),
                np.linalg.norm(p2a - p1b), np.linalg.norm(p2a - p2b),
                np.linalg.norm(mid_a - mid_b),
            )
            if d < dist_th:
                pairs.append((a, b))
    return pairs


def load_rt_csv(path: str) -> dict[int, tuple]:
    """rt_results.csv を読み込み {frame_idx: (R, t)} を返す。success_flag==0 の行はスキップ。"""
    poses = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success_flag"]) == 0:
                continue
            frame = int(row["frame"])
            R = np.array([
                [float(row["R00"]), float(row["R01"]), float(row["R02"])],
                [float(row["R10"]), float(row["R11"]), float(row["R12"])],
                [float(row["R20"]), float(row["R21"]), float(row["R22"])],
            ], dtype=np.float64)
            t = np.array([float(row["t0"]), float(row["t1"]), float(row["t2"])], dtype=np.float64)
            poses[frame] = (R, t)
    return poses


def build_pipeline_context(cfg: dict, K_mat: Optional[np.ndarray], img_size: tuple[int, int]) -> PipelineContext:
    """projector / pose_estimator / line_matcher 等を初期化し PipelineContext を返す。
    projection.enable=False または stage<3 の場合は中身が None/空の Context を返す。"""
    stage = cfg.get("stage", 4)
    proj_cfg = cfg["projection"]
    lm_cfg = cfg["line_matching"]
    prop_cfg = cfg["matching"].get("propagation_filter", {})

    projector = pose_estimator = line_matcher = None
    lines3d: list = []
    confusable_pairs: list[tuple[int, int]] = []
    rt_poses: dict[int, tuple] = {}

    if proj_cfg.get("enable", False) and stage >= 3 and K_mat is not None:
        lines3d_path = proj_cfg.get("lines3d_csv", "")
        if not lines3d_path:
            print("[WARN] projection.lines3d_csv が未設定のため投影をスキップします。", file=sys.stderr)
        else:
            try:
                lines3d = load_lines3d_csv(lines3d_path)
                fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
                projector = LineProjector(fx, fy, cx, cy, lines3d, img_size)
                print(f"[INFO] 3D線分を {len(lines3d)} 本読み込みました: {lines3d_path}")
                confusable_pairs = _compute_confusable_pairs(lines3d)
                if confusable_pairs:
                    print(f"[INFO] 近接する紛らわしい3D線分ペア（矛盾時に両方入れ替え）: {confusable_pairs}")
            except Exception as e:
                print(f"[WARN] lines3d 読み込み失敗: {e}", file=sys.stderr)

        marker_jpg = proj_cfg.get("marker_jpg", "")
        marker_dat = proj_cfg.get("marker_dat", "")
        if projector and marker_jpg and marker_dat:
            try:
                pose_estimator = MarkerPoseEstimator(
                    marker_jpg,
                    marker_dat,
                    ratio=proj_cfg.get("marker_ratio", 0.75),
                    min_matches=proj_cfg.get("marker_min_matches", 8),
                )
                print(f"[INFO] マーカー姿勢推定を初期化: {marker_jpg}")
            except Exception as e:
                print(f"[WARN] MarkerPoseEstimator 初期化失敗: {e}", file=sys.stderr)

        if projector and pose_estimator:
            line_matcher = LineMatcher2D3D(
                angle_th_deg=float(lm_cfg.get("angle_th_deg", 10.0)),
                dist_th=float(lm_cfg.get("dist_th", 60.0)),
                overlap_th=float(lm_cfg.get("overlap_th", 0.2)),
            )
            print(
                f"[INFO] 線分マッチング (LineMatcher2D3D) を初期化: "
                f"angle={lm_cfg['angle_th_deg']}°, dist={lm_cfg['dist_th']}px, "
                f"overlap={lm_cfg['overlap_th']}"
            )

        rt_csv_path = proj_cfg.get("rt_csv", "")
        if rt_csv_path:
            try:
                rt_poses = load_rt_csv(rt_csv_path)
                print(f"[INFO] rt_csv 読み込み: {rt_csv_path}  ({len(rt_poses)} フレーム)")
            except Exception as e:
                print(f"[WARN] rt_csv 読み込み失敗: {e}", file=sys.stderr)

    def get_pose(frame_idx: int):
        """姿勢取得の優先順位: rt_csv > マーカー推定"""
        if frame_idx in rt_poses:
            return rt_poses[frame_idx]
        return None

    return PipelineContext(
        projector=projector,
        pose_estimator=pose_estimator,
        line_matcher=line_matcher,
        lines3d=lines3d,
        confusable_pairs=confusable_pairs,
        get_pose=get_pose,
        K_mat=K_mat,
        stage=stage,
        min_line_corr=int(lm_cfg.get("min_lines", 3)),
        marker_init_frames=int(proj_cfg.get("marker_init_frames", -1)),
        max_rot_jump_deg=float(lm_cfg.get("max_rot_jump_deg", -1.0)),
        max_trans_jump=float(lm_cfg.get("max_trans_jump", -1.0)),
        lm_cfg=lm_cfg,
        prop_enable=bool(prop_cfg.get("enable", False)),
        prop_angle=float(prop_cfg.get("angle_th_deg", 10.0)),
        prop_dist=float(prop_cfg.get("dist_th_px", 100.0)),
        prop_ratio=float(prop_cfg.get("length_ratio_th", 2.0)),
    )


def acquire_pose(ctx: PipelineContext, frame_idx: int, gray: np.ndarray):
    """姿勢取得の優先順位: rt_csv > マーカー推定（初期フレームのみ）。"""
    pose = ctx.get_pose(frame_idx)
    use_marker = ctx.marker_init_frames < 0 or frame_idx < ctx.marker_init_frames
    if pose is None and ctx.pose_estimator is not None and use_marker:
        pose = ctx.pose_estimator.estimate_pose(gray, ctx.K_mat)
    return pose


def seed_initial_2d3d(ctx: PipelineContext, pose, kl) -> dict[int, tuple]:
    """最初のフレームの 2D-3D 対応を、姿勢投影+幾何マッチングで初期化する。"""
    result: dict[int, tuple] = {}
    if ctx.stage >= 4 and pose is not None and ctx.projector is not None and ctx.line_matcher is not None:
        proj_init0 = ctx.projector.project_lines(*pose)
        for proj, _kl, kl_idx in ctx.line_matcher.match(proj_init0, kl):
            result[kl_idx] = (proj.p1_3d, proj.p2_3d)
    return result


# ─────────────────────────────────────────────
# フレーム間で引き継ぐ状態
# ─────────────────────────────────────────────


@dataclass
class TrackingState:
    """フレーム間で引き継ぐ状態（毎フレーム更新される）。"""
    frame: np.ndarray
    gray: np.ndarray
    kl: list
    desc: Optional[np.ndarray]
    pose: Optional[tuple]
    corr_2d3d: dict[int, tuple]


# ─────────────────────────────────────────────
# 1フレーム分の対応付け・姿勢推定処理
# ─────────────────────────────────────────────


@dataclass
class FrameResult:
    curr_2d3d: dict[int, tuple]
    pose_curr: Optional[tuple]
    tracked_kl_indices: set[int]
    tracked_prev_indices: set[int]
    geom_kl_indices: set[int]
    outlier_kl_indices: set[int]
    # 可視化用（RANSAC/MAGSAC除去前のスナップショット。除去された対応も
    # correspondence_only 等で外れ値色として描画し続けるために必要）
    tracked_kl_indices_vis: set[int]
    geom_kl_indices_vis: set[int]
    line3d_labels: dict[int, int]
    proj_prev: list
    proj_curr: list
    corr_diag_rows: list[dict] = field(default_factory=list)


def process_frame(
    ctx: PipelineContext,
    frame_idx: int,
    matches,
    prev_state: TrackingState,
    curr_kl,
    curr_gray: np.ndarray,
) -> FrameResult:
    """1フレーム分の 2D-3D 対応追跡・姿勢推定を行う。

    Stage2: LBD フレーム間対応の 2D-3D 伝播
    Stage3: 姿勢取得 + 3D投影 + 幾何マッチング + 紛らわしい3D線分ペアの矛盾チェック（入れ替え）
    Stage4: GOOPPnPL による姿勢精密化 + RANSAC/MAGSAC 外れ値の追跡状態からの除去
    """
    stage = ctx.stage
    prev_kl = prev_state.kl
    prev_2d3d = prev_state.corr_2d3d
    pose_prev = prev_state.pose

    # ── Stage 4: LBD による 2D-3D 対応の伝播 ─────────────────────
    # 伝播条件: (1) LBD 対応あり (2) 検出線分同士の 2D-2D 幾何整合
    #           （角度差・中点距離・長さ比。姿勢に依存しない）
    curr_2d3d: dict[int, tuple] = {}
    tracked_kl_indices: set[int] = set()
    tracked_prev_indices: set[int] = set()
    if stage >= 4:
        for m in matches:
            if m.queryIdx not in prev_2d3d:
                continue
            if ctx.prop_enable and not is_match_consistent(
                    prev_kl[m.queryIdx], curr_kl[m.trainIdx],
                    ctx.prop_angle, ctx.prop_dist, ctx.prop_ratio):
                continue
            curr_2d3d[m.trainIdx] = prev_2d3d[m.queryIdx]
            tracked_kl_indices.add(m.trainIdx)
            tracked_prev_indices.add(m.queryIdx)

    # ── Stage 3/4: 3D投影 + 幾何マッチング + GOOPPnPL ────────────
    pose_curr: tuple | None = None
    proj_prev: list = []
    proj_curr: list = []
    # RANSAC/MAGSAC で外れ値として除外された kl_idx（可視化・ログ用）
    outlier_kl_indices: set[int] = set()
    geom_2d3d: dict[int, tuple] = {}
    line3d_labels: dict[int, int] = {}
    corr_diag_rows: list[dict] = []

    if stage >= 3:
        # Step 1: 姿勢取得（rt_csv > マーカー推定(初期フレームのみ) > 前フレーム引継ぎ）
        pose_curr = acquire_pose(ctx, frame_idx, curr_gray)

        # Step 2: 幾何学的マッチングで curr_2d3d を補完（幾何由来のみ geom_2d3d に記録）
        ref_pose = pose_curr if pose_curr is not None else pose_prev
        if ctx.projector is not None and ctx.line_matcher is not None and ref_pose is not None:
            proj_init = ctx.projector.project_lines(*ref_pose)
            # LBD 引き継ぎ済みの 3D 線分は幾何マッチング対象から除外
            already_matched = {
                (tuple(p1), tuple(p2)) for p1, p2 in curr_2d3d.values()
            }
            proj_init_filtered = [
                pl for pl in proj_init
                if (tuple(pl.p1_3d), tuple(pl.p2_3d)) not in already_matched
            ]
            for proj, _kl, kl_idx in ctx.line_matcher.match(proj_init_filtered, curr_kl):
                geom_2d3d[kl_idx] = (proj.p1_3d, proj.p2_3d)
                if kl_idx not in curr_2d3d:
                    curr_2d3d[kl_idx] = (proj.p1_3d, proj.p2_3d)

        # ── 近接する紛らわしい3D線分ペアの矛盾チェック ──────────────
        # idx0/23 のように実距離が数cmしかない平行な3D線分ペアは、検出
        # ノイズのせいでどちらがどちらに対応するか入れ替わることがある。
        # 現在の推定姿勢でペア両方を投影した変位ベクトル（画面上, X/Y）と、
        # 実際に割り当てられた検出線分同士の変位ベクトルが逆向き（内積が
        # 負）の場合、2つの対応が入れ替わっていると判断し、割り当てられた
        # 3D線分を交換する（除去はしない）。X方向オフセットのペアだけで
        # なくZ方向（高さ）オフセットのペア（例: idx9/29）にも対応する
        # ため、画面X軸だけでなくX/Y両方を使う。
        if ctx.confusable_pairs and ref_pose is not None and curr_2d3d:
            R_ref_c, t_ref_c = ref_pose
            idx_to_kl: dict[int, int] = {}
            for _kl_idx, (_p1, _p2) in curr_2d3d.items():
                _l3_idx = ctx.lines3d_idx(_p1, _p2)
                if _l3_idx >= 0:
                    idx_to_kl[_l3_idx] = _kl_idx

            def _project_mid(p1, p2, _R=R_ref_c, _t=t_ref_c):
                mid = (np.asarray(p1) + np.asarray(p2)) / 2.0
                Xc = _R @ mid + _t
                if Xc[2] <= 0.0:
                    return None
                u = ctx.K_mat[0, 0] * (Xc[0] / Xc[2]) + ctx.K_mat[0, 2]
                v = ctx.K_mat[1, 1] * (Xc[1] / Xc[2]) + ctx.K_mat[1, 2]
                return float(u), float(v)

            to_swap: list[tuple[int, int]] = []
            for _i, _j in ctx.confusable_pairs:
                kl_i, kl_j = idx_to_kl.get(_i), idx_to_kl.get(_j)
                if kl_i is None or kl_j is None or kl_i == kl_j:
                    continue
                proj_i = _project_mid(*ctx.lines3d[_i])
                proj_j = _project_mid(*ctx.lines3d[_j])
                if proj_i is None or proj_j is None:
                    continue
                det_i = ((curr_kl[kl_i].startPointX + curr_kl[kl_i].endPointX) / 2.0,
                         (curr_kl[kl_i].startPointY + curr_kl[kl_i].endPointY) / 2.0)
                det_j = ((curr_kl[kl_j].startPointX + curr_kl[kl_j].endPointX) / 2.0,
                         (curr_kl[kl_j].startPointY + curr_kl[kl_j].endPointY) / 2.0)
                v_expected = (proj_i[0] - proj_j[0], proj_i[1] - proj_j[1])
                v_actual = (det_i[0] - det_j[0], det_i[1] - det_j[1])
                dot = v_expected[0] * v_actual[0] + v_expected[1] * v_actual[1]
                if dot < 0.0:
                    to_swap.append((kl_i, kl_j))

            for kl_i, kl_j in to_swap:
                curr_2d3d[kl_i], curr_2d3d[kl_j] = curr_2d3d[kl_j], curr_2d3d[kl_i]

        # Step 3 (Stage 4 のみ): GOOPPnPL で姿勢精密化
        if stage >= 4:
            R_init_pose = pose_curr if pose_curr is not None else pose_prev
            if R_init_pose is not None and len(curr_2d3d) >= ctx.min_line_corr:
                R_init, _ = R_init_pose
                corr_for_pose = [
                    (ProjectedLine(p1_3d=p1, p2_3d=p2,
                                   pt1_2d=(curr_kl[idx].startPointX, curr_kl[idx].startPointY),
                                   pt2_2d=(curr_kl[idx].endPointX,   curr_kl[idx].endPointY)),
                     curr_kl[idx], idx)
                    for idx, (p1, p2) in curr_2d3d.items()
                ]
                try:
                    refined, inlier_rel_idx = estimate_from_lines(
                        R_init, corr_for_pose, ctx.K_mat,
                        min_lines=ctx.min_line_corr,
                        sigma_max_px=float(ctx.lm_cfg.get("sigma_max_px", 20.0)),
                        ransac_thresh_px=float(ctx.lm_cfg.get("ransac_thresh_px", 10.0)),
                        robust_method=ctx.lm_cfg.get("robust_method", "magsac"),
                        y_down_constraint=bool(ctx.lm_cfg.get("y_down_constraint", True)),
                        return_inliers=True,
                        min_sample_angle_deg=float(ctx.lm_cfg.get("min_sample_angle_deg", 15.0)))
                    if refined is not None and inlier_rel_idx is not None:
                        inlier_kl_set = {corr_for_pose[i][2] for i in inlier_rel_idx}
                        outlier_kl_indices = set(curr_2d3d.keys()) - inlier_kl_set
                    # ── 姿勢ジャンプゲート ──────────────────────
                    # 前フレーム姿勢からの回転・カメラ位置の跳びが物理的に
                    # あり得ない大きさなら棄却する（反転解・破綻解の入口対策）。
                    if refined is not None and ctx.max_rot_jump_deg > 0:
                        R_new, t_new = refined
                        R_old, t_old = R_init_pose
                        if _rotation_angle_deg(R_new, R_old) > ctx.max_rot_jump_deg:
                            refined = None
                        elif ctx.max_trans_jump > 0:
                            c_new = -R_new.T @ np.asarray(t_new)
                            c_old = -R_old.T @ np.asarray(t_old)
                            if float(np.linalg.norm(c_new - c_old)) > ctx.max_trans_jump:
                                refined = None
                    if refined is not None:
                        pose_curr = refined
                except Exception:
                    pass

        if pose_curr is None:
            pose_curr = pose_prev

        if ctx.projector is not None:
            if pose_prev is not None:
                proj_prev = ctx.projector.project_lines(*pose_prev)
            if pose_curr is not None:
                proj_curr = ctx.projector.project_lines(*pose_curr)

        # ── 2D-3D対応ログ: どの検出線分がどの3D線分に対応したか ──────
        # line3d_labels: 可視化(correspondence_only)で番号表示するための kl_idx -> lines3d.csv行番号
        if ctx.projector is not None and pose_curr is not None:
            R_c, t_c = pose_curr
            for kl_idx, (p1, p2) in curr_2d3d.items():
                kl = curr_kl[kl_idx]
                src = "LBD" if kl_idx in tracked_kl_indices else ("GEOM" if kl_idx in geom_2d3d else "?")
                l3_idx = ctx.lines3d_idx(p1, p2)
                line3d_labels[kl_idx] = l3_idx
                proj_tmp = ProjectedLine(p1_3d=p1, p2_3d=p2, pt1_2d=(0.0, 0.0), pt2_2d=(0.0, 0.0))
                corr_diag_rows.append({
                    "frame_idx": frame_idx,
                    "kl_idx": kl_idx,
                    "line3d_idx": l3_idx,
                    "source": src,
                    "kl_sx": round(float(kl.startPointX), 2),
                    "kl_sy": round(float(kl.startPointY), 2),
                    "kl_ex": round(float(kl.endPointX), 2),
                    "kl_ey": round(float(kl.endPointY), 2),
                    "kl_length": round(float(kl.lineLength), 2),
                    "p1_3d_x": p1[0], "p1_3d_y": p1[1], "p1_3d_z": p1[2],
                    "p2_3d_x": p2[0], "p2_3d_y": p2[1], "p2_3d_z": p2[2],
                    "residual_px": round(_line_residual(proj_tmp, kl, R_c, t_c, ctx.K_mat), 3),
                    "robust_outlier": kl_idx in outlier_kl_indices,
                })

    # 可視化用スナップショット（除去前）: correspondence_only 等で
    # RANSAC/MAGSAC が外れ値として除外した線分も青色で描画し続けるため、
    # 除去前の tracked/geom 集合をここで控えておく。
    tracked_kl_indices_vis = set(tracked_kl_indices)
    geom_kl_indices_vis = set(geom_2d3d.keys()) - tracked_kl_indices if stage >= 3 else set()

    # ── RANSAC/MAGSACで外れ値判定された対応を追跡状態からも除去 ──
    # outlier_kl_indices はそのフレームの姿勢計算用に判定したものだが、
    # ここで curr_2d3d/geom_2d3d/tracked_kl_indices からも除去しないと
    # 次フレームの prev_2d3d にそのまま引き継がれ、同じ誤対応が
    # 毎フレーム残り続けてしまう（corr_diag への記録は上のループで
    # robust_outlier=True として済んでいるため、ここで消しても良い）。
    for _kl_idx in outlier_kl_indices:
        curr_2d3d.pop(_kl_idx, None)
        geom_2d3d.pop(_kl_idx, None)
        tracked_kl_indices.discard(_kl_idx)

    geom_kl_indices = set(geom_2d3d.keys()) - tracked_kl_indices if stage >= 3 else set()

    return FrameResult(
        curr_2d3d=curr_2d3d,
        pose_curr=pose_curr,
        tracked_kl_indices=tracked_kl_indices,
        tracked_prev_indices=tracked_prev_indices,
        geom_kl_indices=geom_kl_indices,
        outlier_kl_indices=outlier_kl_indices,
        tracked_kl_indices_vis=tracked_kl_indices_vis,
        geom_kl_indices_vis=geom_kl_indices_vis,
        line3d_labels=line3d_labels,
        proj_prev=proj_prev,
        proj_curr=proj_curr,
        corr_diag_rows=corr_diag_rows,
    )
