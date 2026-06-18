/* ************************************************************ OIPnPL.c *** *
 * 点対応と線対応からカメラの位置姿勢を推定する関数
 *
 * Copyright (C) 2023-2025 Yasuyuki SUGAYA <sugaya.yasuyuki.jp@tut.jp>
 *
 *                                 Time-stamp: <2025-02-04 08:25:05 sugaya>
 * ************************************************************************* */
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>

/* 行列からベクトルへの変換 ************************************************ */
static Eigen::VectorXd
R_mat_to_vec (const Eigen::Matrix3d&	R) {
  Eigen::VectorXd r(9);
  r <<
    R(0, 0), R(0, 1), R(0, 2),
    R(1, 0), R(1, 1), R(1, 2),
    R(2, 0), R(2, 1), R(2, 2);
  return r;
}

/* 並進ベクトルの計算 ****************************************************** */
static void
calc_t (const Eigen::Matrix3d&	R,
	const Eigen::MatrixXd&	t_param,
	Eigen::Vector3d&	t) {
  Eigen::VectorXd r = R_mat_to_vec (R);
  t = t_param * r;
}

/* 並進ベクトルの計算に必要な行列の計算 ************************************ */
static Eigen::MatrixXd
calc_t_param (const std::vector<Eigen::Matrix3d>&	V,
	      const std::vector<Eigen::Vector3d>&	P) {
  Eigen::Matrix3d sum1 = Eigen::Matrix3d::Zero();
  Eigen::MatrixXd sum2 = Eigen::MatrixXd::Zero(3, 9);
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity ();

  for (int n = 0; n < V.size(); n++) {
    sum1 += (I - V[n]);
    
    Eigen::MatrixXd work = Eigen::MatrixXd::Zero (3, 9);
    work(0, 0) = work(1, 3) = work(2, 6) = P[n](0);
    work(0, 1) = work(1, 4) = work(2, 7) = P[n](1);
    work(0, 2) = work(1, 5) = work(2, 8) = P[n](2);

    sum2 += (V[n] - I) * work;
  }
  return sum1.inverse() * sum2;
}

/* 目的関数の値計算　******************************************************* */
static double
calc_J (const std::vector<Eigen::Vector3d>&	P,
	const std::vector<Eigen::Matrix3d>&	V,
	const Eigen::Matrix3d&			R,
	const Eigen::Vector3d&			t) {
  double J = 0.0;
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  for (int n = 0; n < P.size(); n++) {
    J += ((I - V[n]) * (R * P[n] + t)).squaredNorm();
  }
  return J / P.size();
}

/* スケールパラメータの計算 ************************************************ */
static double
calc_scale_parameter (const std::vector<Eigen::Vector3d>&	P,
		      const Eigen::Matrix3d&			R,
		      const Eigen::Vector3d&			t) {
  Eigen::Vector3d Pc = Eigen::Vector3d::Zero();
  for (int n = 0; n < P.size(); n++) Pc += (R * P[n] + t);
  Pc /= P.size();
  
  return Pc.norm();
}

/* 線対応と点対応からカメラ運動を計算する関数 ****************************** */
double
OIPnPL (const std::vector<Eigen::Vector3d>&	P,
	const std::vector<Eigen::Matrix3d>&	V,
	const std::vector<Eigen::Vector3d>&	Pd,
	const std::vector<Eigen::Matrix3d>&	Vd,	
	Eigen::Matrix3d&			R,
	Eigen::Vector3d&			t,
	std::array<int, 3> use_flag,	
	int					scale_flag,
	double					eps) {

  /* 並進ベクトルの計算に必要なベクトル計算 */
  Eigen::MatrixXd t_param = calc_t_param (V, P);

  /* 並進ベクトルの計算 */
  calc_t (R, t_param, t);

  /* 3次元点座標のスケーリング */
  std::vector<Eigen::Vector3d> Ps = P;
  if (scale_flag) {
    /* スケールパラメータの計算 */
    double scale = calc_scale_parameter (P, R, t);
    for (int n = 0; n < Ps.size(); n++) Ps[n] /= scale;
  }
  /* ３次元点の重心計算、重心を原点に並行移動した３次元点の計算 */
  Eigen::Vector3d Pc = Eigen::Vector3d::Zero();
  for (int n = 0; n < Ps.size(); n++)  Pc += Ps[n];
  Pc /= Ps.size();
  std::vector<Eigen::Vector3d> P_ = Ps;
  for (int n = 0; n < P_.size(); n++)  P_[n] = Ps[n] - Pc;

  /* 目的関数の値の計算 */
  double J = calc_J (Ps, V, R, t);
  
  /* カメラ運動の計算(メインループ) **************************************** */
  int niterations = 0;  
  while (1) {
    /* 共分散行列の計算 */
    Eigen::Vector3d Pc = Eigen::Vector3d::Zero();
    if (use_flag[0] == 1 || use_flag[1] == 1){
      for (int n = 0; n < Ps.size(); n++) Pc += (V[n] * (R * Ps[n] + t));
      Pc /= Ps.size();
    }

    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    if (use_flag[0] == 1 || use_flag[1] == 1){
      for (int n = 0; n < Ps.size(); n++) {
        S += (V[n] * (R * Ps[n] + t) - Pc) * P_[n].transpose();
      }
    }
    if (use_flag[2] == 1) {
      for (int n = 0; n < Pd.size(); n++) {
	      S += (Vd[n] * R * Pd[n]) * Pd[n].transpose();
      }
    }
    /* 共分散行列の特異値分解 */
    Eigen::JacobiSVD<Eigen::Matrix3d> svd (S,
					   Eigen::ComputeFullU |
					   Eigen::ComputeFullV);
    Eigen::Matrix3d sig = Eigen::Matrix3d::Identity ();
    sig(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();

    /* 回転行列の計算 */
    Eigen::Matrix3d R_ = svd.matrixU() * sig * svd.matrixV().transpose();

    /* 並進ベクトルの計算 */
    calc_t (R_, t_param, t);

    /* 収束判定 */
    double J_ = calc_J (Ps, V, R_, t);
    double err = fabs (J - J_);
    if (err < eps) break;

    niterations++;
    
    R = R_;
    J = J_;
    
    if (niterations > 10000) {
      eps *= 10.0;
      niterations = 0;
    }
  }
  return J;
}

/* ***************************************************** End of OIPnPL.c *** */
