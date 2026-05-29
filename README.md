# line_feature_matching

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

## ディレクトリ構成

```
project/
├── src/
│   ├── main.py        # CLI・設定読み込み・処理パイプライン
│   ├── detector.py    # 線分検出・向き正規化・記述子計算
│   ├── matcher.py     # Lowe's ratio test によるマッチング
│   └── writer.py      # 可視化描画・CSV書き込み
├── config/
│   └── config.json    # 設定ファイル
├── video/
│   └── (入力動画をここに置く)
└── output/            # 実行時に自動作成
    ├── <stem>_line_match.mp4
    ├── <stem>_matches.csv
    └── <stem>_line_match_frames/
        ├── frame_00001.png
        └── ...
```

---

## 使い方

### 基本的な実行

プロジェクトルートから以下のコマンドで実行します。

```bash
python src/main.py --config config/config.json
```

`--config` を省略した場合は `config/config.json` が使用されます。

```bash
python src/main.py
```

---

## config.json リファレンス

すべてのパラメータは `config/config.json` で指定します。未指定のキーはデフォルト値で補完されます。

```json
{
    "io": {
        "input":        "video/16.mp4",
        "output":       "",
        "csv":          "",
        "fallback_fps": 30.0
    },
    "detection": {
        "detector":         "lsd",
        "max_lines":        200,
        "resize_width":     0,
        "lsd_octave_scale": 2,
        "lsd_num_octaves":  1
    },
    "descriptor": {
        "lsd_scale":          0.8,
        "edlines_min_length": 30.0
    },
    "matching": {
        "ratio_thresh": 0.75
    },
    "visualization": {
        "max_draw":        60,
        "only_matched":    false,
        "line_thickness":  2,
        "match_thickness": 1,
        "text_font_scale": 0.6,
        "text_thickness":  1
    },
    "geometry": {
        "vertical_epsilon": 1e-4
    }
}
```

### `io` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `input` | （必須） | 入力動画のパス（例: `video/16.mp4`） |
| `output` | `output/<入力名>_line_match.mp4` | 出力動画のパス（空欄で自動生成） |
| `csv` | `output/<入力名>_matches.csv` | 対応情報CSVのパス（空欄で自動生成） |
| `fallback_fps` | `30.0` | FPS取得失敗時のフォールバック値 |

### `detection` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `detector` | `"lsd"` | 線分検出器の選択（`"lsd"` または `"edlines"`） |
| `max_lines` | `200` | フレームあたりの最大検出線分数 |
| `resize_width` | `0` | 処理前にリサイズする幅（px）。`0` で変更なし |
| `lsd_octave_scale` | `2` | LSD検出時のピラミッドスケール |
| `lsd_num_octaves` | `1` | LSD検出時のオクターブ数 |

### `descriptor` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `lsd_scale` | `0.8` | LSDParam のスケール |
| `edlines_min_length` | `30.0` | EDLines使用時の最小線分長（px） |

### `matching` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `ratio_thresh` | `0.75` | Lowe's ratio test の閾値（小さいほど厳しい） |

### `visualization` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `max_draw` | `60` | 可視化に描画するマッチ数の上限 |
| `only_matched` | `false` | 対応が取れた線分のみ描画するフラグ |
| `line_thickness` | `2` | 検出線分の描画太さ（px） |
| `match_thickness` | `1` | 対応線の描画太さ（px） |
| `text_font_scale` | `0.6` | 情報テキストのフォントサイズ |
| `text_thickness` | `1` | 情報テキストの線幅 |

### `geometry` セクション

| キー | デフォルト | 説明 |
|---|---|---|
| `vertical_epsilon` | `1e-4` | 垂直線判定の許容誤差（ε） |

---

## 出力ファイル

すべての出力は `output/` ディレクトリに保存されます。`output/` は実行時に自動作成されます。
`io.output` / `io.csv` を空欄にした場合、入力動画のステム名から以下のパスが自動生成されます。

```
output/
├── <stem>_line_match.mp4          # 出力動画
├── <stem>_matches.csv             # 対応情報CSV
└── <stem>_line_match_frames/      # フレーム画像フォルダ
    ├── frame_00001.png
    ├── frame_00002.png
    └── ...
```

出力先を明示したい場合は `io.output` / `io.csv` にパスを指定してください。親ディレクトリが存在しない場合も自動で作成されます。

### 動画（mp4）

横並び2フレームに検出線分と対応線を描画した動画。

- 左フレーム：緑の線分（前フレーム）
- 右フレーム：青の線分（現フレーム）
- オレンジの線：対応線（両フレームの線分中点を結ぶ）
- 左上に検出線分数・マッチ数のテキスト表示

### フレーム画像（`<stem>_line_match_frames/`）

各フレームペアの可視化画像を PNG で保存します。

| ファイル名 | 内容 |
|---|---|
| `frame_XXXXX.png` | 対応線つきの可視化画像 |

### CSV

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

LSD固有のパラメータは `detection` および `descriptor` セクションで調整できます。

| パラメータ | キー | 目安 | 効果 |
|---|---|---|---|
| ピラミッドスケール | `lsd_octave_scale` | `1`〜`4` | 大きいほど粗い解像度で検出 |
| オクターブ数 | `lsd_num_octaves` | `1`〜`3` | 大きいほど多スケールで検出 |
| LSDスケール | `lsd_scale` | `0.5`〜`1.0` | 大きいほど細かい線まで検出 |

### EDLines

`cv2.ximgproc.createEdgeDrawing()` を使用します。LSDより高速ですが、短い線分が多く検出される傾向があります。
**向いているシーン**: リアルタイム処理が必要な場合、またはエッジが豊富なシーン

長い線分を優先したい場合は `edlines_min_length` を大きくしてください。

| パラメータ | キー | 目安 | 効果 |
|---|---|---|---|
| 最小線分長 | `edlines_min_length` | `30`〜`100` | 大きくすると短い線が除外される |

---

## 主要な内部処理

### 線分の向き正規化（`normalize_line_directions`）

LBD記述子は線分の向きに依存するため、始点・終点が逆向きに検出されるとミスマッチが増えます。本ツールでは全線分を「左→右（垂直なら上→下）」に統一することで、この問題を軽減しています。垂直判定の許容誤差は `geometry.vertical_epsilon` で調整できます。

### Lowe's ratio test（`match_lines`）

kNN（k=2）で候補マッチを取得し、最近傍距離が2番目の近傍距離の `ratio_thresh` 倍未満のものだけを採用します。値を小さくすると厳しいマッチングになり偽対応が減りますが、マッチ数も減ります。

---

## パフォーマンスのヒント

高解像度映像（1080p以上）では処理が重くなりがちです。以下の対処が有効です。

- `resize_width: 960` などでリサイズして処理する
- `max_lines: 100` で検出数を抑える
- `detector: "edlines"` に切り替える（LSDより高速）
- フレームレートの高い動画は事前に間引きしておく

---

## 注意事項

- `cv2.line_descriptor` および `cv2.ximgproc` はOpenCVのcontribモジュールに含まれます。`opencv-contrib-python` が必須です。
- EDLinesのPythonバインディングでは `EdgeDrawing_Params` による詳細なパラメータ設定ができない場合があります（バージョン依存）。
- 出力動画は `mp4v` コーデックを使用します。環境によっては再生できない場合があるため、その場合はffmpegでH.264に変換してください。

```bash
ffmpeg -i output/<stem>_line_match.mp4 -vcodec libx264 output/<stem>_line_match_h264.mp4
```