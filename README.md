# line_feature_matching.py

映像の各フレームから線特徴を検出し、隣接フレーム間で対応付けを行うツールです。

---

## 概要

本ツールは動画の連続フレーム間で線分（エッジ）を検出・マッチングし、その結果を動画・画像・CSVとして出力します。建物・道路・人工物など直線的な構造物を含むシーンでの映像解析、カメラ運動推定、シーン変化検出などへの応用を想定しています。

### 処理パイプライン

```
入力動画
  │
  ▼
フレーム読み込み（逐次）
  │
  ▼
グレースケール変換
  │
  ▼
線分検出（LSD または EDLines）
  │
  ▼
LBD記述子の計算（BinaryDescriptor）
  │
  ▼
隣接フレーム間でマッチング（Lowe's ratio test）
  │
  ▼
可視化フレームの生成・保存
  │
  ▼
出力動画 / PNG画像 / CSV
```

---

## 機能一覧

| 機能 | 内容 |
|---|---|
| 線分検出 | LSD（Line Segment Detector）または EDLines から選択可能 |
| 記述子計算 | LBD（Line Binary Descriptor）による特徴量記述 |
| マッチング | BinaryDescriptorMatcher + Lowe's ratio test |
| 向き正規化 | 線分の向きを「左→右」に統一してミスマッチを抑制 |
| 可視化 | 横並び2フレーム＋対応線の描画 |
| 出力 | 動画（mp4）、フレーム画像（PNG）、対応情報（CSV） |

---

## 依存環境

- Python 3.8 以上
- OpenCV（`opencv-contrib-python` が必要）

```bash
pip install opencv-contrib-python numpy
```

> **注意**: `opencv-python`（contrib なし）では `cv2.line_descriptor` および `cv2.ximgproc` が使えません。必ず `opencv-contrib-python` をインストールしてください。

---

## 使い方

### 基本的な実行

```bash
# LSD検出器（デフォルト）
python line_feature_matching.py --input video.mp4

# EDLines検出器を使用
python line_feature_matching.py --input video.mp4 --detector edlines

# 出力先を明示的に指定
python line_feature_matching.py --input video.mp4 --output result.mp4 --csv matches.csv
```

### オプション一覧

| オプション | 省略形 | デフォルト | 説明 |
|---|---|---|---|
| `--input` | `-i` | （必須） | 入力動画のパス |
| `--output` | `-o` | `<入力名>_line_match.mp4` | 出力動画のパス |
| `--csv` | `-c` | `<入力名>_matches.csv` | 対応情報CSVのパス |
| `--max_lines` | | `200` | フレームあたりの最大検出線分数 |
| `--ratio_thresh` | | `0.75` | Lowe's ratio test の閾値（小さいほど厳しい） |
| `--max_draw` | | `60` | 可視化に描画するマッチ数の上限 |
| `--resize_width` | | `0`（変更なし） | 処理前にリサイズする幅（px） |
| `--detector` | | `lsd` | 線分検出器の選択（`lsd` または `edlines`） |
| `--only_matched` | | `False` | 対応が取れた線分のみを描画するフラグ |

### 実行例

```bash
# 解像度を半分にしてEDLinesで処理
python line_feature_matching.py --input video.mp4 --detector edlines --resize_width 960

# 検出数を増やしてマッチングを厳しくする
python line_feature_matching.py --input video.mp4 --max_lines 300 --ratio_thresh 0.65

# 対応線のみ描画（ノイズ抑制）
python line_feature_matching.py --input video.mp4 --only_matched
```

---

## 出力ファイル

### 動画（mp4）

横並び2フレームに検出線分と対応線を描画した動画。

- 左フレーム：緑の線分（前フレーム）
- 右フレーム：青の線分（現フレーム）
- オレンジの線：対応線（両フレームの線分中点を結ぶ）
- 右上にフレーム番号・検出数・マッチ数のテキスト表示

### フレーム画像（output/ フォルダ）

各フレームペアを2種類のPNG画像として保存します。

| ファイル名 | 内容 |
|---|---|
| `frame_XXXXX.png` | 対応線つきの可視化画像 |
| `frame_XXXXX_lines.png` | 検出線分のみの画像（対応線なし） |

### CSV（matches.csv）

フレームペアごとのマッチング結果を1行1マッチで記録します。

| カラム名 | 説明 |
|---|---|
| `frame_idx` | フレーム番号（1始まり） |
| `match_rank` | フレーム内でのマッチ順位（0始まり） |
| `distance` | Hamming距離（小さいほど類似） |
| `kl1_sx`, `kl1_sy` | 前フレーム線分の始点座標 |
| `kl1_ex`, `kl1_ey` | 前フレーム線分の終点座標 |
| `kl1_angle` | 前フレーム線分の角度（ラジアン） |
| `kl1_length` | 前フレーム線分の長さ（px） |
| `kl2_*` | 現フレームの対応線分（同上） |

---

## 検出器の選択

### LSD（デフォルト）

OpenCVの `LSDDetector` を使用します。精度が高く、長い線分を安定して検出できます。
**向いているシーン**: 建物・インフラ・人工構造物など直線が明確な映像

### EDLines

`cv2.ximgproc.createEdgeDrawing()` を使用します。LSDより高速ですが、短い線分が多く検出される傾向があります。**向いているシーン**: リアルタイム処理が必要な場合、またはエッジが豊富なシーン

#### EDLinesのパラメータ調整（コード内）

長い線分を優先したい場合、`detect_and_describe` 内の以下の値を変更してください。

```python
MIN_LENGTH = 30.0  # この値を大きくすると短い線が除外される（例：80.0）
```

| パラメータ | 目安 | 効果 |
|---|---|---|
| `MinPathLength` | 100〜200 | エッジチェーンの最小画素数（大きいほど短いエッジを除外） |
| `MinLineLength` | 30〜60 | 線分として認める最小長（px） |
| `PFmode` | `True` | 隣接する短い線分を長い線にまとめる |
| `NFAValidation` | `True` | 偽検出を統計的に抑制 |
| `MIN_LENGTH` | 50〜100 | 最終的な長さフィルタ（解像度に合わせて調整） |

---

## 主要な内部処理

### 線分の向き正規化（`normalize_line_directions`）

LBD記述子は線分の向きに依存するため、始点・終点が逆向きに検出されるとミスマッチが増えます。本ツールでは全線分を「左→右（垂直なら上→下）」に統一することで、この問題を軽減しています。

### Lowe's ratio test（`match_lines`）

kNN（k=2）で候補マッチを取得し、最近傍距離が2番目の近傍距離の `ratio_thresh` 倍未満のものだけを採用します。値を小さくすると厳しいマッチングになり偽対応が減りますが、マッチ数も減ります。

---

## パフォーマンスのヒント

高解像度映像（1080p以上）では処理が重くなりがちです。以下の対処が有効です。

- `--resize_width 960` などでリサイズして処理する
- `--max_lines 100` で検出数を抑える
- `--detector edlines` に切り替える（LSDより高速）
- フレームレートの高い動画は事前に間引きしておく

---

## 注意事項

- `cv2.line_descriptor` および `cv2.ximgproc` はOpenCVのcontribモジュールに含まれます。`opencv-contrib-python` が必須です。
- EDLinesのPythonバインディングでは `EdgeDrawing_Params` による詳細なパラメータ設定ができない場合があります（バージョン依存）。
- 出力動画は `mp4v` コーデックを使用します。環境によっては再生できない場合があるため、その場合はffmpegでH.264に変換してください。

```bash
ffmpeg -i output.mp4 -vcodec libx264 output_h264.mp4
```