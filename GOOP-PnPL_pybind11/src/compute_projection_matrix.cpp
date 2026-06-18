/* ******************************************* compute_projectin_matrix.cpp ***
 * 点特徴と線特徴の射影行列を計算する関数
 *
 * ************************************************************************* */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>
#include "compute_projection_matrix.hpp"

/* 点特徴の射影行列を計算 */
void
compute_point_projection_matrix(
    std::vector<Eigen::Vector3d>& i_P,
    std::vector<Eigen::Matrix3d>& V) {
    for (int n = 0; n < i_P.size(); n++) {
        Eigen::Vector3d p = i_P[n].normalized();
        V.push_back (p * p.transpose());
    }
}

/* 線特徴の射影行列を計算 */
void
compute_line_projection_matrix(
    std::vector<Eigen::Vector3d>& i_P,
    std::vector<Eigen::Matrix3d>& V) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    for (int n = 0; n+1 < i_P.size(); n+=2) {
        Eigen::Vector3d p1 = i_P[n].normalized();
        Eigen::Vector3d p2 = i_P[n+1].normalized();
        Eigen::Vector3d nv = p1.cross (p2);
        V.push_back (I - nv * nv.transpose());
    }
}