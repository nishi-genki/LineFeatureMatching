import numpy as np
import sys
import os

sys.path.append("../build")
import GOOPPnPL


# データ読み込み
Pp = np.loadtxt("sample_data/point3d.dat")
i_Pp = np.loadtxt("sample_data/point2d.dat")
Pl = np.loadtxt("sample_data/line3d.dat")
i_Pl = np.loadtxt("sample_data/line2d.dat")
print(len(Pp))
print(len(i_Pp))
print(len(Pl))
print(len(i_Pl))

use_flag = [1, 1, 1]

# 推定
R1, t1, R2, t2 = GOOPPnPL.GOOPPnPL_main(Pp, i_Pp, Pl, i_Pl, use_flag)

Rt = np.loadtxt("sample_data/Rt.dat")

print("推定値：\n")
print("R1 =\n", R1)
print("t1 =", t1)
print("R2 =\n", R2)
print("t2 =", t2)
print("真値：")
print("Rt =\n", Rt)
