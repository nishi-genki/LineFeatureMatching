/* ************************************************************ OIPnPL.h *** *
 * 点対応と線対応からカメラの位置姿勢を推定する関数 ヘーダファイル
 *
 * Copyright (C) 2023-2025 Yasuyuki SUGAYA <sugaya.yasuyuki.jp@tut.jp>
 *
 *                                 Time-stamp: <2025-02-03 17:07:17 sugaya>
 * ************************************************************************* */
#ifndef	__OIPnPL_H__
#define	__OIPnPL_H__

double
OIPnPL (const std::vector<Eigen::Vector3d>&	P,
	const std::vector<Eigen::Matrix3d>&	V,
	const std::vector<Eigen::Vector3d>&	Pd,
	const std::vector<Eigen::Matrix3d>&	Vd,	
	Eigen::Matrix3d&			R,
	Eigen::Vector3d&			t,
	std::array<int, 3> use_flag,
	int					scale_flag,
	double					eps);

#endif	/* __OIPnPl_H__ */

/* ***************************************************** End of OIPnPL.h *** */
