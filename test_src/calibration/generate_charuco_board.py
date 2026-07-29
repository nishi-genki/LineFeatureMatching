"""
generate_charuco_board.py
==========================
印刷用のChArUcoボード画像を生成する。

ChArUcoボードはチェッカーボード＋ArUcoマーカーの組み合わせで、
盤の一部が画角外や隠れていても検出できるため、通常のチェッカーボードより
画角の端・隅を撮影しやすく、キャリブレーションの網羅性を確保しやすい。

内部パラメータ(fx,fy,cx,cy,dist)の校正では、マス目の物理サイズは結果に
影響しない（スケールの不定性は外部パラメータ(rvec/tvec)のみに現れる）ため、
squares_x/squares_y の升目数と印刷サイズだけ気にすればよい。A4用紙に収まる
構成をデフォルトにしている。

使い方:
    python3 test_src/calibration/generate_charuco_board.py
    → test_src/calibration/charuco_board.png を生成
"""

import cv2

# ── ボード仕様 ────────────────────────────────────────────
SQUARES_X = 7          # 横方向のマス数
SQUARES_Y = 5          # 縦方向のマス数
SQUARE_LEN_M = 0.035    # マス目の一辺（校正結果には影響しない。印刷レイアウト用）
MARKER_LEN_M = 0.026    # ArUcoマーカーの一辺（SQUARE_LEN_M の70~80%が目安）
DICT_NAME = "DICT_4X4_50"

PX_PER_M = 12000        # 印刷解像度 ≈ 300dpi相当（実寸はプリンタ側の等倍印刷設定で調整する）


def main():
    dictionary = getattr(cv2.aruco, DICT_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary)

    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LEN_M, MARKER_LEN_M, dictionary,
    )

    img_w = int(SQUARES_X * SQUARE_LEN_M * PX_PER_M)
    img_h = int(SQUARES_Y * SQUARE_LEN_M * PX_PER_M)
    img = board.generateImage((img_w, img_h), marginSize=40, borderBits=1)

    out_path = "test_src/calibration/charuco_board.png"
    cv2.imwrite(out_path, img)
    print(f"生成: {out_path}  ({img_w}x{img_h}px)")
    print(f"実寸: {SQUARES_X * SQUARE_LEN_M * 100:.1f}cm x {SQUARES_Y * SQUARE_LEN_M * 100:.1f}cm"
          f"  (マス目一辺 {SQUARE_LEN_M*100:.1f}cm)")
    print("\n印刷時は「実際のサイズ」「等倍」で印刷し、拡大縮小がかからないよう注意。")
    print("印刷後、上記のマス目一辺の実測値を CALIBRATE_VIDEO.md 等に控えておくと、")
    print("後で別プロジェクトで外部パラメータのスケールが必要になった際に使える"
          "（今回の内部パラメータ校正自体には不要）。")


if __name__ == "__main__":
    main()
