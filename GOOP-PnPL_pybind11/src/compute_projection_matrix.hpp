/* ******************************************* compute_projectin_matrix.hpp ***
 * 点特徴と線特徴の射影行列を計算する関数　ヘッダファイル
 *
 * ************************************************************************* */
#ifndef	__COMPUTE_PROJECTION_MATRIX_HPP__
#define	__COMPUTE_PROJECTION_MATRIX_HPP__

#include <vector>
#include <Eigen/Dense>

/* 点特徴の射影行列計算 */
void
compute_point_projection_matrix(
    std::vector<Eigen::Vector3d>& P,
    std::vector<Eigen::Matrix3d>& V);

/* 線特徴の射影行列計算 */
void
compute_line_projection_matrix(
    std::vector<Eigen::Vector3d>& P,
    std::vector<Eigen::Matrix3d>& V);

#endif /* __COMPUTE_PROJECTION_MATRIX_HPP__ */

/* ********************************* End of compute_projectin_matrix.hpp *** */