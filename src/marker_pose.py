"""
marker_pose.py
==============
画像マーカーによるカメラ姿勢推定。
C++ DetectMarker + ImageMarker + PoseEstimator::ComputeCameraPose_p の Python移植。

  - 特徴点検出: AKAZE (C++ と同一)
  - マッチング:  BFMatcher(NORM_HAMMING) + Lowe's ratio test (C++ と同一)
  - 姿勢推定:   cv2.solvePnPRansac (C++ GOPPnPL の代替)

.dat フォーマット:
    4行、各行タブまたはスペース区切りの x y z
    順序: [左上, 右上, 右下, 左下]  (C++ coords[0..3])
"""

from typing import Optional

import cv2
import numpy as np


def _load_dat(dat_path: str) -> list[np.ndarray]:
    """maker-XX.dat の4頂点 3D 座標を読み込む。"""
    coords: list[np.ndarray] = []
    with open(dat_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords.append(np.array(list(map(float, line.split())), dtype=np.float64))
    if len(coords) != 4:
        raise ValueError(f"dat ファイルは4行必要です (読み込み: {len(coords)}行): {dat_path}")
    return coords


class MarkerPoseEstimator:
    """
    画像マーカーから各フレームのカメラ姿勢 (R, t) を推定する。

    3D 座標の線形補間式 (C++ ImageMarker と同一):
        world = origin + mX * (px/W) + mY * (py/H)
        origin = coords[0], mX = coords[1]-coords[0], mY = coords[3]-coords[0]
    """

    def __init__(
        self,
        marker_jpg: str,
        marker_dat: str,
        ratio: float = 0.75,
        min_matches: int = 8,
    ):
        img = cv2.imread(marker_jpg, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"マーカー画像を開けません: {marker_jpg}")

        self._h, self._w = img.shape[:2]
        self._ratio      = ratio
        self._min_matches = min_matches

        coords = _load_dat(marker_dat)
        self._origin = coords[0]
        self._mX     = coords[1] - coords[0]   # 横方向 (C++ mX)
        self._mY     = coords[3] - coords[0]   # 縦方向 (C++ mY)

        # C++ cv::AKAZE::create() + cv::BFMatcher(cv::NORM_HAMMING)
        self._detector = cv2.AKAZE_create()
        self._matcher  = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._kps, self._descs = self._detector.detectAndCompute(img, None)

    def estimate_pose(
        self,
        frame_gray: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        フレームからマーカーを検出し、カメラ姿勢 (R 3x3, t 3-vec) を返す。
        マーカーが見つからない場合は None。
        """
        kps, descs = self._detector.detectAndCompute(frame_gray, None)
        if descs is None or len(kps) < self._min_matches:
            return None

        # knn マッチング + ratio test (C++ と同一)
        knn = self._matcher.knnMatch(self._descs, descs, k=2)
        good = [m for m, n in knn if m.distance < self._ratio * n.distance]
        if len(good) < self._min_matches:
            return None

        # マーカー上の画素座標 → 3D 世界座標 (C++ ImageMarker::feature_match と同一)
        obj_pts: list[np.ndarray] = []
        img_pts: list[tuple[float, float]] = []
        for m in good:
            px, py = self._kps[m.queryIdx].pt
            world = self._origin + self._mX * (px / self._w) + self._mY * (py / self._h)
            obj_pts.append(world)
            img_pts.append(kps[m.trainIdx].pt)

        obj_arr = np.array(obj_pts, dtype=np.float32)
        img_arr = np.array(img_pts, dtype=np.float32)

        K64   = K.astype(np.float64)
        dist64 = dist.astype(np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, K64, dist64,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.99,
        )
        if not ok or inliers is None or len(inliers) < self._min_matches:
            return None

        # Levenberg-Marquardt による精密化 (GOPPnPL の反復最適化に相当)
        inlier_obj = obj_arr[inliers.flatten()].astype(np.float64)
        inlier_img = img_arr[inliers.flatten()].astype(np.float64)
        rvec, tvec = cv2.solvePnPRefineLM(inlier_obj, inlier_img, K64, dist64, rvec, tvec)

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()
