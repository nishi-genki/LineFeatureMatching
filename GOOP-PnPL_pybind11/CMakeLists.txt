cmake_minimum_required(VERSION 3.4)
project("GOOPPnPL" CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SRC_FILES
    src/GOOPPnPL_main.cpp
    src/GOOPPnPL.cpp
    src/OIPnPL.cpp
    src/solver_pnpl.cpp
    src/compute_projection_matrix.cpp
)

pybind11_add_module(GOOPPnPL ${SRC_FILES})