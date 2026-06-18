/* *********************************************************** GOPPnPL.c *** *
 * プログラムの説明
 *
 * Copyright (C) 2025 Yasuyuki SUGAYA <sugaya.yasuyuki.jp@tut.jp>
 *
 *                                 Time-stamp: <2025-09-10 12:18:31 sugaya>
 * ************************************************************************* */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "solver_pnpl.hpp"
#include "GOOPPnPL.hpp"

/* ************************************************************************* */
static Eigen::VectorXd
R_mat_to_vec (const Eigen::Matrix3d&	R) {
  Eigen::VectorXd r(9);
  r <<
    R(0, 0), R(0, 1), R(0, 2),
    R(1, 0), R(1, 1), R(1, 2),
    R(2, 0), R(2, 1), R(2, 2);
  return r;
}

/* ************************************************************************* */
static void
calc_A (const std::vector<Eigen::Vector3d>&	P,
	std::vector<Eigen::MatrixXd>&		A) {
  Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero (3, 9);
  for (size_t n = 0; n < P.size(); n++) {
    A_(0, 0) = A_(1, 3) = A_(2, 6) = P[n](0);
    A_(0, 1) = A_(1, 4) = A_(2, 7) = P[n](1);
    A_(0, 2) = A_(1, 5) = A_(2, 8) = P[n](2);
    A.push_back (A_);
  }
}

/* ************************************************************************* */
static void
calc_B (const std::vector<Eigen::Matrix3d>&	V,
	const std::vector<Eigen::MatrixXd>&	A,
	Eigen::MatrixXd&			B) {
  Eigen::MatrixXd work1 = Eigen::MatrixXd::Zero(3, 3);
  Eigen::MatrixXd work2 = Eigen::MatrixXd::Zero(3, 9);
  for (size_t n = 0; n < V.size(); n++) {
    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(3, 3) - V[n];
    work1 += K;
    work2 += (K * A[n]);
  }
  B = -work1.inverse() * work2;
}

/* ************************************************************************* */
static void
calc_D (const std::vector<Eigen::Matrix3d>&	V,
	const std::vector<Eigen::MatrixXd>&	A,
	const Eigen::MatrixXd&			B,
	const std::vector<Eigen::Matrix3d>&	Vd,
	const std::vector<Eigen::MatrixXd>&	C,
	Eigen::MatrixXd&			D,
	std::array<int, 3> flag) {
  Eigen::MatrixXd D1 = Eigen::MatrixXd::Zero(9, 9);
  Eigen::MatrixXd D2 = Eigen::MatrixXd::Zero(9, 9);
  if (!A.empty()) {
    for (size_t n = 0; n < A.size(); n++) {
      Eigen::MatrixXd K = Eigen::MatrixXd::Identity(3, 3) - V[n];
      D1 += ((A[n] + B).transpose() * K * (A[n] + B));
    }
  }
  if (!C.empty()) {
    for (size_t n = 0; n < C.size(); n++) {
      Eigen::MatrixXd K = Eigen::MatrixXd::Identity(3, 3) - Vd[n];
      D2 += (C[n].transpose() * K * C[n]);
    }
  }
  D = D1 + D2;
}

/* ************************************************************************* */
static Eigen::Matrix3d
calc_R (const Eigen::VectorXcd&	r) {
  double a = r(0).real();
  double b = r(1).real();
  double c = r(2).real();
  double q_norm_2 = ((1 + a * a + b * b + c * c));
  Eigen::Matrix3d R;
  R <<
    1 + a * a - b * b - c * c,
    2 * (a * b - c),
    2 * (a * c + b),
    2 * (a * b + c),
    1 - a * a + b * b - c * c,
    2 * (b * c - a),
    2 * (a * c - b),
    2 * (b * c + a),
    1 - a * a - b * b + c * c;
  R /= q_norm_2;

  return R;
} 

/* ************************************************************************* */
static Eigen::Vector3d
calc_t (const Eigen::Matrix3d&	R,
	const Eigen::MatrixXd&	B) {
  Eigen::VectorXd r = R_mat_to_vec (R);
  return B * r;
}

/* ************************************************************************* */
static double
calc_J (const Eigen::Matrix3d&	R,
	const Eigen::MatrixXd&	D) {
  Eigen::VectorXd r = R_mat_to_vec (R);
  return r.dot (D * r);
}

/* ************************************************************************* */
static bool
check_real (const Eigen::VectorXcd&	x) {
  for (int n = 0; n < 3; n++) {
    if (fabs (x(n).imag()) > 1e-6) {
      return false;
    }
  }
  return true;
}

/* ************************************************************************* */
bool
check_position (const Eigen::Matrix3d&			R,
		const Eigen::Vector3d&			t,
		const std::vector<Eigen::Vector3d>&	Pp,
		const std::vector<Eigen::Vector3d>&	Pl) {
  double z_ave = 0.0;
  for (size_t n = 0; n < Pp.size(); n++) {
    Eigen::Vector3d p = R * Pp[n] + t;
    z_ave += p(2);
  }
  for (size_t n = 0; n < Pl.size(); n++) {
    Eigen::Vector3d p = R * Pl[n] + t;
    z_ave += p(2);
  }
  z_ave /= Pp.size() + Pl.size();

  return (z_ave > 0) ? true : false;
}

/* 線対応と点対応からカメラ運動を計算する関数 ****************************** */
bool
GOPPnPL (const std::vector<Eigen::Vector3d>&	Pp,
	 const std::vector<Eigen::Matrix3d>&	Vp,
	 const std::vector<Eigen::Vector3d>&	Pl,
	 const std::vector<Eigen::Matrix3d>&	Vl,
	 const std::vector<Eigen::Vector3d>&	Pd,
	 const std::vector<Eigen::Matrix3d>&	Vd,
	 std::vector<Solution>&			solution,
	 std::array<int, 3> use_flag,
	 bool					output_flag,
	 bool					verbose) {

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
    for (size_t n = 0; n < Pl.size(); n++)  {
      P.push_back (Pl[n]);
      V.push_back (Vl[n]);
    }
  }
  std::vector<Eigen::MatrixXd> A;  
  Eigen::MatrixXd B(3, 9);
  if (use_flag[0] == 1 || use_flag[1] == 1) {
    calc_A (P, A);
    calc_B (V, A, B);
  }

  std::vector<Eigen::MatrixXd> C;
  if (use_flag[2] == 1) calc_A (Pd, C);
  
  std::vector<Eigen::MatrixXd> A_;
  Eigen::MatrixXd B_(3, 9);
  if (use_flag[0] == 0 && use_flag[1] == 0 && use_flag[2] == 1) {
    calc_A (P, A_);
    calc_B (V, A_, B_);
  }
  Eigen::MatrixXd D(9, 9);
  calc_D (V, A, B, Vd, C, D, use_flag);

  std::vector<Eigen::VectorXcd> candidate_solution = solve (D);

  double min_J = 1e+10;
  Solution min_sol;
  
  for (Eigen::VectorXcd x : candidate_solution) {
    Solution sol;
    sol.R = calc_R (x);
    if (use_flag[0] == 1 || use_flag[1] == 1) {
      sol.t = calc_t (sol.R, B);
    } else {
      sol.t = calc_t (sol.R, B_);
    }
    sol.J       = calc_J (sol.R, D);
    sol.is_real = check_real (x);
    if (sol.J < min_J) {
      min_J = sol.J;
      min_sol = sol;
    }
    solution.push_back (sol);
  }
  std::sort (solution.begin(), solution.end(),
	     [](const Solution& a, const Solution& b) {
	       return a.J < b.J;
	     });
  return min_sol.is_real;
}

/* **************************************************** End of GOIPnPL.c *** */
