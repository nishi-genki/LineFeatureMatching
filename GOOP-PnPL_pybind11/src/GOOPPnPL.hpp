/* *********************************************************** GOPPnPL.h *** *
 * ヘーダファイル
 *
 * Copyright (C) 2023-2024 Yasuyuki SUGAYA <sugaya.yasuyuki.jp@tut.jp>
 *
 *                                 Time-stamp: <2025-09-10 10:34:31 sugaya>
 * ************************************************************************* */
#ifndef	__GOPPnPL_H__
#define	__GOPPnPL_H__

class Solution {
 public:
  Solution() {
    R       = Eigen::Matrix3d::Zero();
    t       = Eigen::Vector3d::Zero();
    J       = 1.0e+10;
    is_real = false;
  };
  Eigen::Matrix3d	R;
  Eigen::Vector3d	t;
  double		J;
  bool			is_real;
};

bool
GOPPnPL (const std::vector<Eigen::Vector3d>&	Pp,
	 const std::vector<Eigen::Matrix3d>&	Vp,
	 const std::vector<Eigen::Vector3d>&	Pl,
	 const std::vector<Eigen::Matrix3d>&	Vl,	
	 const std::vector<Eigen::Vector3d>&	Pd,
	 const std::vector<Eigen::Matrix3d>&	Vd,
	 std::vector<Solution>&			solution,	 
	 std::array<int, 3> 	use_flag,
	 bool					output_flag,
	 bool					verbose);

void
output_all_solution (const std::vector<Solution>& solution);

bool
check_position (const Eigen::Matrix3d&			R,
		const Eigen::Vector3d&			t,
		const std::vector<Eigen::Vector3d>&	Pp,
		const std::vector<Eigen::Vector3d>&	Pl);

#endif	/* __GOPPnPl_H__ */

/* **************************************************** End of GOPPnPL.h *** */
