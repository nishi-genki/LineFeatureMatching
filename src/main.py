"""
main.py
=======
CLI エントリポイント・設定読み込み。

映像処理パイプライン本体は pipeline.py に、2D-3D対応追跡・姿勢推定ロジックは
tracking.py に、診断ログの集約・出力は diagnostics.py に分離されている。
このファイルは設定読込とCLI引数処理のみを担う。

使い方:
    python src/main.py --config config/config.json
"""

import argparse
import json
import sys
from pathlib import Path
import time

import pipeline

# ─────────────────────────────────────────────
# 設定読み込み
# ─────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """
    JSON 設定ファイルを読み込み、default_config.json のデフォルト値で補完して返す。
    default_config.json は config_path と同じディレクトリに置く。
    """

    def deep_merge(a: dict, b: dict) -> dict:
        """b の値で a を再帰的に上書きして返す（非破壊）。"""
        out = {**a}
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    default_path = Path(config_path).parent / "default_config.json"
    with open(default_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    cfg = deep_merge(defaults, user_cfg)

    # 入力必須チェック
    if not cfg["io"].get("input"):
        print("[ERROR] config.json の io.input を指定してください。", file=sys.stderr)
        sys.exit(1)

    # 出力パスの自動生成（動画は output/video/ にまとめる）
    output_dir = Path("output")
    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(cfg["io"]["input"]).stem
    if not cfg["io"].get("output"):
        cfg["io"]["output"] = str(video_dir / f"{stem}_line_match.mp4")
    if not cfg["io"].get("csv"):
        cfg["io"]["csv"] = str(output_dir / f"{stem}_matches.csv")

    return cfg


def _primary_output_path(cfg: dict) -> Path:
    """診断ログ・設定保存先の基準となる出力パスを返す。
    visualization.mode が複数指定（リスト）の場合は pipeline.process_video 側の
    _mode_output_path と同じ規則（先頭モードのファイル名に _<モード名> を付加）で揃える。"""
    out_file = Path(cfg["io"]["output"])
    vis_mode_cfg = cfg.get("visualization", {}).get("mode", "side_by_side")
    vis_modes = [vis_mode_cfg] if isinstance(vis_mode_cfg, str) else list(vis_mode_cfg)
    if len(vis_modes) > 1:
        out_file = out_file.with_name(f"{out_file.stem}_{vis_modes[0]}{out_file.suffix}")
    return out_file


def store_config(cfg: dict, name: str = "config_used.json") -> Path:
    """
    読み込んだ cfg をフレーム画像と同じフォルダに保存して Path を返す。
    例: "output/edlines_parameter.mp4" -> "output/edlines_parameter_frames/config_used.json"
    """
    out_file = _primary_output_path(cfg)
    out_dir = Path("output") / (out_file.stem + "_frames")  # フレーム保存先と同じフォルダ
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    try:
        out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 設定を保存 → {out_path}")
    except Exception as e:
        print(f"[WARN] 設定保存に失敗: {e}", file=sys.stderr)
    return out_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="映像から線特徴を検出しフレーム間対応付けを行う (LSD + LBD or EDLines)")
    p.add_argument(
        "--config",
        "-c",
        default="config/config.json",
        help="設定ファイルパス (default: config/config.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 乱数シードを決定して cfg に埋め込む（config_used.json に保存される）
    # io.use_fixed_seed が true の場合は io.random_seed の値を使い、false の場合は時刻から生成する
    io_cfg_seed = cfg.get("io", {})
    if io_cfg_seed.get("use_fixed_seed", False):
        seed = io_cfg_seed.get("random_seed", 0)
    else:
        seed = int(time.time() * 1000) % (2 ** 32)
    cfg["io"]["random_seed"] = seed
    print(f"[INFO] random_seed: {seed} (fixed={io_cfg_seed.get('use_fixed_seed', False)})")

    # 読み込んだ設定を出力先フォルダへ保存
    store_config(cfg)

    start = time.perf_counter()
    pipeline.process_video(cfg)
    elapsed = time.perf_counter() - start

    print(f"[TIME] 全処理時間: {elapsed:.3f}秒")


if __name__ == "__main__":
    main()
