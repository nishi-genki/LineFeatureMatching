"""
projector.py
============
3次元線分を画像平面へ投影する (C++ LineProjector / Lines3D の Python移植)。

lines3d CSV フォーマット: x1,y1,z1,x2,y2,z2  (1行1線分)
poses CSV フォーマット:    frame_idx,r11,r12,r13,r21,r22,r23,r31,r32,r33,tx,ty,tz
  * frame_idx は 0-based
  * # で始まる行はコメントとして無視
"""

import csv
import math
from typing import NamedTuple, Optional

import cv2
import numpy as np


class ProjectedLine(NamedTuple):
    """1本の3D線分とその2D投影。line_matcher / line_pose が参照する。"""
    p1_3d: np.ndarray          # 3D 端点 1 (world)
    p2_3d: np.ndarray          # 3D 端点 2 (world)
    pt1_2d: tuple[float, float]  # 投影 2D 端点 1 (px)
    pt2_2d: tuple[float, float]  # 投影 2D 端点 2 (px)


def load_lines3d_csv(csv_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            vals = list(map(float, row))
            lines.append((
                np.array(vals[0:3], dtype=np.float64),
                np.array(vals[3:6], dtype=np.float64),
            ))
    return lines


def load_poses_csv(csv_path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            vals = list(map(float, row))
            frame_idx = int(vals[0])
            R = np.array(vals[1:10], dtype=np.float64).reshape(3, 3)
            t = np.array(vals[10:13], dtype=np.float64)
            poses[frame_idx] = (R, t)
    return poses


class LineProjector:
    """
    3次元線分を画像平面へ投影する。
    C++ LineProjector::projectLines と同一のピンホール+歪み補正式を使用する。
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: list[float],
        lines3d: list[tuple[np.ndarray, np.ndarray]],
        img_size: tuple[int, int],
    ):
        self._K = np.array([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self._dist = np.array(dist, dtype=np.float64).flatten()
        self._lines3d = lines3d
        self._img_size = img_size  # (width, height)

    def project_lines(
        self,
        R: np.ndarray,
        t: np.ndarray,
    ) -> list[ProjectedLine]:
        """
        全3D線分を投影し、画像内にクリップされた ProjectedLine リストを返す。
        C++ LineProjector::projectLines と同等。
        """
        result: list[ProjectedLine] = []
        for p1_w, p2_w in self._lines3d:
            pts2d = []
            valid = True
            for Pw in (p1_w, p2_w):
                Xc = R @ Pw + t
                Z = Xc[2]
                if Z <= 0.0:
                    valid = False
                    break
                u, v = self._project_point(Xc[0] / Z, Xc[1] / Z)
                pts2d.append((u, v))

            if valid and len(pts2d) == 2:
                clipped = self._clip_to_image(pts2d[0], pts2d[1])
                if clipped is not None:
                    result.append(ProjectedLine(
                        p1_3d=p1_w,
                        p2_3d=p2_w,
                        pt1_2d=clipped[0],
                        pt2_2d=clipped[1],
                    ))
        return result

    # ──────────────────────────────────────────
    # 内部処理
    # ──────────────────────────────────────────

    def _project_point(self, x: float, y: float) -> tuple[float, float]:
        """正規化座標 (x, y) に歪み補正とカメラ行列を適用して画素座標を返す。
        C++ と同一の式: radial + tangential distortion。"""
        d = self._dist
        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        p1 = d[2] if len(d) > 2 else 0.0
        p2 = d[3] if len(d) > 3 else 0.0
        k3 = d[4] if len(d) > 4 else 0.0

        r2 = x * x + y * y
        radial  = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        u = self._K[0, 0] * x_d + self._K[0, 2]
        v = self._K[1, 1] * y_d + self._K[1, 2]
        return u, v

    def _clip_to_image(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """C++ isSegmentInImage + cv::clipLine 相当。
        nan/inf・int32範囲外の値を安全に処理する。"""
        u1, v1 = float(p1[0]), float(p1[1])
        u2, v2 = float(p2[0]), float(p2[1])
        if not all(math.isfinite(v) for v in (u1, v1, u2, v2)):
            return None
        w, h = self._img_size
        # OpenCV は int32 相当の座標しか受け付けないのでクランプ
        _CLAMP = 1 << 20
        def _ci(v: float) -> int:
            return int(max(-_CLAMP, min(_CLAMP, v)))
        pt1 = (_ci(u1), _ci(v1))
        pt2 = (_ci(u2), _ci(v2))
        ok, c1, c2 = cv2.clipLine((0, 0, w, h), pt1, pt2)
        if ok:
            return (float(c1[0]), float(c1[1])), (float(c2[0]), float(c2[1]))
        return None
