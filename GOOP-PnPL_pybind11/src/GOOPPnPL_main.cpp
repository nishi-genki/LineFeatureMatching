/* ************************************************************** main.c *** *
 * プログラムの説明
 *
 * Copyright (C) 2024 Yasuyuki SUGAYA <sugaya.yasuyuki.jp@tut.jp>
 *
 *                                    Time-stamp: <2025-09-10 10:16:55 sugaya>
 * ************************************************************************* */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "compute_projection_matrix.hpp"
#include "GOOPPnPL.hpp"
#include "OIPnPL.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

/* ************************************************************************* */
static Eigen::Matrix3d
R_from_quoternion (const Eigen::Vector4d&	l) {
  double a = l(0);
  double b = l(1);
  double c = l(2);
  double d = l(3);
  Eigen::Matrix3d R;
  R <<
    a * a + b * b - c * c - d * d,
    2 * (b * c - a * d),
    2 * (b * d + a * c),
    2 * (b * c + a * d),
    a * a - b * b + c * c - d * d,
    2 * (c * d - a * b),
    2 * (b * d - a * c),
    2 * (c * d + a * b),
    a * a - b * b - c * c + d * d;
  return R;
}

/* ************************************************************************* *
 * X軸回りの回転行列
 * ************************************************************************* */
static Eigen::Matrix3d
rotation_x (double angle) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R(1, 1) = cos(angle);
  R(1, 2) =-sin(angle);
  R(2, 1) = sin(angle);
  R(2, 2) = cos(angle);

  return R;
}

/* ************************************************************************* *
 * Y軸回りの回転行列
 * ************************************************************************* */
static Eigen::Matrix3d
rotation_y (double angle) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R(0, 0) = cos(angle);
  R(0, 2) = sin(angle);
  R(2, 0) =-sin(angle);
  R(2, 2) = cos(angle);

  return R;
}

/* ************************************************************************* *
 * Z軸回りの回転行列
 * ************************************************************************* */
static Eigen::Matrix3d
rotation_z (double angle) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R(0, 0) = cos(angle);
  R(0, 1) =-sin(angle);
  R(1, 0) = sin(angle);
  R(1, 1) = cos(angle);

  return R;
}

/* ************************************************************************* */
void
generate_rotation_mat (std::mt19937&	engine,
		     Eigen::Matrix3d&	        R) {
  std::uniform_real_distribution<> generator (-M_PI, M_PI);
  double ax = generator(engine);
  double ay = generator(engine);
  double az = generator(engine);

  R = rotation_z(az) * rotation_y(ay) * rotation_x(ax);
}
  
/* ************************************************************************* */
static void
rotate_data (const Eigen::Matrix3d&		R,
	     std::vector<Eigen::Vector3d>&	P) {
  for (int n = 0; n < (int) P.size(); n++) P[n] = R * P[n];
}

/* ************************************************************************* */
static Eigen::Matrix3d
create_rotation (void) {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  std::uint32_t seed = (std::uint32_t)
    std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  std::mt19937 engine (seed);
  Eigen::Matrix3d R;
#if 0
  std::uniform_real_distribution<> generator (-1, 1);
  Eigen::Vector4d l = Eigen::Vector4d::Zero();
  l(0) = 0.0;
  l(1) = generator (engine);
  l(2) = generator (engine);
  l(3) = generator (engine);
  l.normalize();
  R = R_from_quoternion (l);
#else
  std::uniform_real_distribution<> generator (-M_PI, M_PI);
  double ax = generator(engine);
  double ay = generator(engine);
  double az = generator(engine);
  R = rotation_z(az) * rotation_y(ay) * rotation_x(ax);
#endif
  return R;
}

/* データの重心が原点になるように平行移動 ********************************** */
static void
shift_data (std::vector<Eigen::Vector3d>&	P,
	    Eigen::Vector3d&			C) {
  C = Eigen::Vector3d::Zero();
  for (int n = 0; n < (int) P.size(); n++) C += P[n];
  C /= P.size();
  for (int n = 0; n < (int) P.size(); n++) P[n] = P[n] - C;
}

/* メイン関数 ************************************************************** */
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>
GOOPPnPL_main (
  std::vector<Eigen::Ref<const Eigen::Vector3d>>& Pp_,   // 3次元点の座標（点特徴）
  std::vector<Eigen::Ref<const Eigen::Vector3d>>& i_Pp_, // 2次元点の座標（3要素目に焦点距離）
  std::vector<Eigen::Ref<const Eigen::Vector3d>>& Pl_,   // 3次元線分の座標（始点, 終点, 始点, 終点, ..の並び）
  std::vector<Eigen::Ref<const Eigen::Vector3d>>& i_Pl_, // 2次元線分の座標（始点, 終点, 始点, 終点, ..の並び、3要素目に焦点距離）
  std::array<int, 3> use_flag // 使用する特徴（点、線（直線上の点）、線（ベクトル））を 0/1 で選択　例：(1, 0, 0) 点特徴のみ使用
) {
  /* データ形式変換 */
  std::vector<Eigen::Vector3d> Pp;
  std::vector<Eigen::Vector3d> i_Pp;
  for (int i = 0; i < Pp_.size(); i++) {
    Pp.push_back(Pp_[i]);
    i_Pp.push_back(i_Pp_[i]);
  }
  std::vector<Eigen::Vector3d> Pl;
  std::vector<Eigen::Vector3d> i_Pl;
  for (int i = 0; i < Pl_.size(); i++) {
    Pl.push_back(Pl_[i]);
    i_Pl.push_back(i_Pl_[i]);
  }

  /* 射影行列の計算 */
  std::vector<Eigen::Matrix3d> Vp;
  std::vector<Eigen::Matrix3d> Vl;
  compute_point_projection_matrix(i_Pp, Vp);
  compute_line_projection_matrix(i_Pl, Vl);

  /* 3次元線分の中点を計算 */
  std::vector<Eigen::Vector3d> Pl_m;
  for (int i = 0; i+1 < Pl.size(); i+=2) {
    Eigen::Vector3d p_m = (Pl[i] + Pl[i+1]) / 2;
    Pl_m.push_back(p_m);
  }

  /* 3次元直線の方向ベクトルの計算 */
  std::vector<Eigen::Vector3d> Pd;
  for (int i = 0; i+1 < Pl.size(); i+=2) {
    Eigen::Vector3d d = Pl[i+1] - Pl[i];
    Pd.push_back(d);
  }

  /* 乱数設定 */
  std::random_device rd;
  std::mt19937 engine(rd());

  /* 有効な解が計算されるまで繰り返す */
  std::vector<Solution> solution;
  bool result = false;
  while(1){
    /* 3次元データをランダムに回転 */
    Eigen::Matrix3d R = create_rotation();
    std::vector<Eigen::Vector3d> Pp_rot = Pp;
    std::vector<Eigen::Vector3d> Pl_m_rot = Pl_m;
    std::vector<Eigen::Vector3d> Pd_rot = Pd;
    rotate_data(R, Pp_rot);
    rotate_data(R, Pl_m_rot);
    rotate_data(R, Pd_rot);

    solution.clear();
    bool result = GOPPnPL (Pp_rot, Vp, Pl_m_rot, Vl, Pd_rot, Vl, solution, use_flag, false, false);

    /* もとに戻す */
    solution[0].R = solution[0].R * R;
    solution[1].R = solution[1].R * R;

    if (result && check_position(solution[0].R, solution[0].t, Pp, Pl_m)) break;
  }

  /* データをまとめる */
  std::vector<Eigen::Vector3d> P;
  std::vector<Eigen::Matrix3d> V;
  if (use_flag[0] == 1) {
    for (size_t n = 0; n < Pp.size(); n++)  {
      P.push_back (Pp[n]);
      V.push_back (Vp[n]);
    }
  }
  if (use_flag[1] == 1 ||
      (use_flag[0] == 0 && use_flag[1] == 0 && use_flag[2] == 1)) {
    for (size_t n = 0; n < Pl_m.size(); n++)  {
      P.push_back (Pl_m[n]);
      V.push_back (Vl[n]);
    }
  }

  /* 大域最適解法で計算した解を初期値として反復解法を適用 */
  Eigen::Matrix3d R1 = solution[0].R;
  Eigen::Vector3d t1;
  double err1 = OIPnPL (P, V, Pd, Vl, R1, t1, use_flag, 0, 1e-12);

  Eigen::Matrix3d R2 = solution[1].R;
  Eigen::Vector3d t2;
  double err2 = OIPnPL (P, V, Pd, Vl, R2, t2, use_flag, 0, 1e-12);
  
  return std::make_tuple(R1, t1, R2, t2);
}

PYBIND11_MODULE(GOOPPnPL, m) {
  m.doc() = "GOOPPnPL pose estimation module";
  m.def("GOOPPnPL_main", &GOOPPnPL_main, "姿勢推定メイン関数");
}

/* ******************************************************* End of main.c *** */
