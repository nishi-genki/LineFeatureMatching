# 大域最適解法 (C++) を Python で使用する

**pybind11** を使って C++ で実装された大域最適解法を Python から利用する方法について、**Windows** と **Linux** の両方で説明します。

---

## 動作環境

- Python 3.x
- CMake 3.4 以上
- Visual Studio 2019 / 2022 または g++ / clang++
- pybind11
- Eigen

---

## インストール手順

### pybind11 のインストール

`pybind11` は pip で簡単にインストールできます。

```sh
pip install pybind11
```

## ビルド方法

環境ごとに C++ プログラムをビルドする必要があります。  
ビルド後、`build` フォルダ内に Windows なら `Release/GOPPnPL*.pyd`、Linux なら `GOPPnPL*.so` が生成されます。

> **注意**  
> Windows では、pip で pybind11 をインストールした場合、`pybind11Config.cmake` が見つからないというエラーが出ることがあります。その際は「トラブルシューティング」をご参照ください。

### Windows

```sh
mkdir build
cd build
cmake .. #ここに-DCMAKE_PREFIX_PATH=="C:\Users\genki\pybind11"を追加
cmake --build . --config Release
```

### Linux
```sh
mkdir build
cd build
cmake ..
make
```

## Python 側での使用方法

テストプログラム（`test/test.py`）も参考にしてください。

### インポート例

```python
import sys
sys.path.append('build/Release')  # Windows の場合
# sys.path.append('build')        # Linux の場合
import GOPPnPL
```
### 関数の呼び出し例

大域最適解法によるカメラ姿勢推定関数 `GOPPnPL_main` が使用できます。

```python
R, t, J = GOPPnPL.GOPPnPL_main(Pp, i_Pp, Pl, i_Pl)
```
#### 引数

- `Pp`: 3次元点の座標（shape: N×3）
- `i_Pp`: `Pp` に対応する2次元点の座標（3要素目に焦点距離、shape: N×3）
- `Pl`: 3次元線分の座標（[始点, 終点, 始点, 終点, ...] の並び、shape: M×3（M/2 が線特徴の数））
- `i_Pl`: `Pl` に対応する2次元線分の座標（同様の並び、始点・終点の3要素目に焦点距離、shape: M×3）

> **補足**  
> - 点特徴のみを使用する場合：`Pl`、`i_Pl` に空のリストを渡してください。
> - 線特徴のみを使用する場合：`Pp`、`i_Pp` に空のリストを渡してください。
>
> 例:
> ```python
> # 点特徴のみ
> R, t, J = GOPPnPL.GOPPnPL_main(Pp, i_Pp, [], [])
>
> # 線特徴のみ
> R, t, J = GOPPnPL.GOPPnPL_main([], [], Pl, i_Pl)
> ```

#### 戻り値

- `R`: 回転行列
- `t`: 並進ベクトル
- `J`: 目的関数の値

## トラブルシューティング

### Windows で `pybind11Config.cmake` が見つからない場合

pip でインストールした pybind11 では CMake 用の設定ファイルが見つからない場合があります。  
この場合は以下の手順で「CMake対応のpybind11」を用意してください。

1. [pybind11 releases](https://github.com/pybind/pybind11/releases) から `pybind11-x.x.x.zip` をダウンロード＆解凍

2. 解凍したフォルダ内で以下を実行

    ```sh
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="C:/local/pybind11"
    cmake --build . --config Release
    cmake --install . --config Release
    ```

    ※ `C:/local/pybind11` は任意のディレクトリでOK

3. ビルド時に `cmake ..` のコマンドに以下を追加

    ```
    -DCMAKE_PREFIX_PATH="C:/local/pybind11"
    ```