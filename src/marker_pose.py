"""
marker_pose.py
==============
画像マーカーによるカメラ姿勢推定。

2つの検出モードを自動判別する（marker_jpg のファイル名から判定）:

  - ArUco モード（ファイル名が "id<N>_DICT_..." に一致する場合）:
      cv2.aruco で4隅を直接検出する。反復パターンの誤対応が原理的に
      起きないため、特徴点マッチングより高信頼。.dat の4隅がそのまま
      2D-3D対応になる。
  - 画像マーカーモード（上記に一致しない場合、従来の実装）:
      C++ DetectMarker + ImageMarker + PoseEstimator::ComputeCameraPose_p
      の Python移植。AKAZE (C++と同一) + BFMatcher(NORM_HAMMING) +
      Lowe's ratio test (C++と同一) + solvePnPRansac でインライア選別。
      ※ ArUco 等の反復的な二値パターンの画像には不向き（局所特徴が
      酷似し誤対応が多発する）。そちらは必ず ArUco モードを使うこと。

姿勢推定: 両モードとも GOOPPnPL (点特徴, use_flag=[1,0,0]) を使用。
コスト最小解 R1, t1 を採用（初期位置推定のため参照姿勢は使わない。
4点の平面配置でも合成データ200試行で反転0件を確認済み）。
GOOPPnPL が使えない場合は solvePnP 系にフォールバック。

歪み補正: フレーム側が undistort 済みである前提のため、ここでは歪み補正
を行わない（dist は None で呼ぶ）。歪み補正前のフレームを渡す場合のみ
dist に係数を渡せば従来通り点単位で補正する。

.dat フォーマット:
    4行、各行タブまたはスペース区切りの x y z
    順序: [左上, 右上, 右下, 左下]  (C++ coords[0..3])
"""

import os
import re
import sys
from typing import Optional

import cv2
import numpy as np

# GOOPPnPL pybind11 モジュールのインポート（line_pose.py と同一手順）
_BIND_DIR = os.path.join(os.path.dirname(__file__),
                         "../GOOP-PnPL_pybind11/build")
sys.path.insert(0, os.path.abspath(_BIND_DIR))
try:
    import GOOPPnPL as _goppnpl
    _GOPPNPL_AVAILABLE = True
except ImportError:
    _GOPPNPL_AVAILABLE = False


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

    # ファイル名 "id<N>_DICT_xxx.png" に一致すれば ArUco モード
    _ARUCO_NAME_RE = re.compile(r"id(\d+)_(DICT_\w+?)(?:\.\w+)?$", re.IGNORECASE)

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
        self._ratio = ratio
        self._min_matches = min_matches

        coords = _load_dat(marker_dat)  # [左上, 右上, 右下, 左下]
        self._obj_pts = np.stack(coords).astype(np.float64)   # ArUco モード用 (4,3)
        self._origin = coords[0]
        self._mX = coords[1] - coords[0]  # 横方向 (C++ mX)　※ 画像マーカーモード用
        self._mY = coords[3] - coords[0]  # 縦方向 (C++ mY)

        self._aruco_detector = None
        self._aruco_id = None
        m = self._ARUCO_NAME_RE.search(os.path.basename(marker_jpg))
        if m is not None:
            marker_id = int(m.group(1))
            dict_name = m.group(2).upper()
            dict_const = getattr(cv2.aruco, dict_name, None)
            if dict_const is not None:
                dictionary = cv2.aruco.getPredefinedDictionary(dict_const)
                params = cv2.aruco.DetectorParameters()
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                detector = cv2.aruco.ArucoDetector(dictionary, params)
                # マーカー画像自身で検出できる（=名前と実体が一致する）ことを確認
                corners, ids, _ = detector.detectMarkers(img)
                if ids is not None and marker_id in ids.flatten():
                    self._aruco_detector = detector
                    self._aruco_id = marker_id

        if self._aruco_detector is None:
            # 画像マーカーモード: C++ cv::AKAZE::create() + cv::BFMatcher(cv::NORM_HAMMING)
            self._detector = cv2.AKAZE_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self._kps, self._descs = self._detector.detectAndCompute(img, None)

    def estimate_pose(
        self,
        frame_gray: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        フレームからマーカーを検出し、カメラ姿勢 (R 3x3, t 3-vec) を返す。
        マーカーが見つからない場合は None。
        dist: 歪み係数。None ならフレームは undistort 済みとして扱う。
        """
        if self._aruco_detector is not None:
            return self._estimate_pose_aruco(frame_gray, K, dist)
        return self._estimate_pose_akaze(frame_gray, K, dist)

    def _estimate_pose_aruco(
        self,
        frame_gray: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """ArUco の4隅を直接検出して姿勢を推定する（誤対応が原理的に無い）。"""
        corners, ids, _ = self._aruco_detector.detectMarkers(frame_gray)
        if ids is None:
            return None
        matches = np.where(ids.flatten() == self._aruco_id)[0]
        if len(matches) == 0:
            return None

        img_pts = corners[matches[0]].reshape(4, 2).astype(np.float64)  # [左上,右上,右下,左下]
        obj_pts = self._obj_pts

        K64    = K.astype(np.float64)
        dist64 = None if dist is None else dist.astype(np.float64)

        pose = self._goppnpl_pose(obj_pts, img_pts, K64, dist64)
        if pose is not None:
            return pose

        # フォールバック: GOOPPnPL が使えない/失敗した場合は平面PnP(IPPE)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K64, dist64, flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K64, dist64, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()

    def _estimate_pose_akaze(
        self,
        frame_gray: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """AKAZE特徴点マッチングで姿勢を推定する（自然画像マーカー用）。"""
        kps, descs = self._detector.detectAndCompute(frame_gray, None)
        if descs is None or len(kps) < self._min_matches:
            return None

        # knn マッチング + ratio test (C++ と同一)
        knn = self._matcher.knnMatch(self._descs, descs, k=2)
        good = [m for m, n in knn if m.distance < self._ratio * n.distance]
        if len(good) < self._min_matches:
            return None

        # マーカー上の画素座標 → 3D 世界座標 (C++ ImageMarker::feature_match と同一)
        Pp: list[np.ndarray] = []
        img_pts: list[tuple[float, float]] = []
        for m in good:
            px, py = self._kps[m.queryIdx].pt
            world = self._origin + self._mX * (px / self._w) + self._mY * (py / self._h)
            Pp.append(world.astype(np.float64))
            img_pts.append(kps[m.trainIdx].pt)

        K64   = K.astype(np.float64)
        dist64 = None if dist is None else dist.astype(np.float64)

        obj_arr = np.array(Pp,      dtype=np.float32)
        img_arr = np.array(img_pts, dtype=np.float32)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr, img_arr, K64, dist64,
            iterationsCount=200,
            reprojectionError=4.0,
            confidence=0.99,
        )
        if not ok or inliers is None or len(inliers) < self._min_matches:
            return None

        inlier_obj = obj_arr[inliers.flatten()].astype(np.float64)
        inlier_img = img_arr[inliers.flatten()].astype(np.float64)

        pose = self._goppnpl_pose(inlier_obj, inlier_img, K64, dist64)
        if pose is not None:
            return pose

        # フォールバック: GOOPPnPL が使えない/失敗した場合は LM 反復
        rvec, tvec = cv2.solvePnPRefineLM(inlier_obj, inlier_img, K64, dist64, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()

    @staticmethod
    def _goppnpl_pose(
        obj_pts: np.ndarray,      # (N,3) インライア3D点
        img_pts: np.ndarray,      # (N,2) インライア画像点
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """GOOPPnPL（点特徴）で姿勢を推定する。失敗時は None。

        コスト最小解 R1, t1 を常に採用する。R1 は C++ 側で点群がカメラ前方に
        来ることを確認済みの大域最適解を OIPnPL で改良したもの。
        """
        if not _GOPPNPL_AVAILABLE:
            return None

        # 正規化座標へ変換（dist が None ならフレーム側で undistort 済みとして
        # K の逆変換のみ行う）。i_Pp は C++ 側で正規化され視線方向としてのみ
        # 使われるため、(x_n, y_n, 1) を渡せば fx≠fy でも厳密。
        und = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), K, dist).reshape(-1, 2)

        Pp   = [obj_pts[i].copy() for i in range(len(obj_pts))]
        i_Pp = [np.array([und[i, 0], und[i, 1], 1.0]) for i in range(len(und))]

        try:
            R1, t1, _R2, _t2 = _goppnpl.GOOPPnPL_main(Pp, i_Pp, [], [], [1, 0, 0])
        except Exception:
            return None

        return (np.asarray(R1, dtype=np.float64),
                np.asarray(t1, dtype=np.float64).flatten())
