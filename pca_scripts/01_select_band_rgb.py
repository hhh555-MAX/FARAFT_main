import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许脚本从任意工作目录启动时仍能导入项目模块。
    sys.path.insert(0, str(ROOT))

from utils.hsi_io import discover_hsi_datasets, load_envi


def load_config(config_path):
    """读取选波段伪 RGB 配置。"""
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def parse_wavelengths(header):
    """从 ENVI 头文件字段中解析波长列表。"""
    wavelengths = header.get("wavelength")
    if wavelengths is None:
        raise ValueError("ENVI 头文件中缺少 wavelength 字段，无法按波长选波段。")
    return np.asarray([float(value) for value in wavelengths], dtype=np.float32)


def nearest_band_index(wavelengths, target_nm):
    """找到最接近目标波长的 0-based 波段下标。"""
    return int(np.argmin(np.abs(wavelengths - float(target_nm))))


def normalize_channel(channel, lower_percentile, upper_percentile):
    """按百分位把单个反射率波段归一化到 0-255。"""
    lo, hi = np.percentile(channel, [lower_percentile, upper_percentile])
    if hi <= lo:
        return np.zeros_like(channel, dtype=np.uint8)
    normalized = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255.0 + 0.5).astype(np.uint8)


def make_band_rgb(cube, wavelengths, config):
    """从高光谱立方体中选取 R/G/B 三个真实波段组成伪 RGB 图。"""
    channels = config["channels"]
    lower = float(config.get("lower_percentile", 1.0))
    upper = float(config.get("upper_percentile", 99.0))

    # 用户关心的波长是 449、548、598 nm；显示时按 RGB 习惯排列为
    # R=598 nm, G=548 nm, B=449 nm。
    channel_order = [
        ("red", channels["red"]),
        ("green", channels["green"]),
        ("blue", channels["blue"]),
    ]
    rgb_channels = []
    selected = []
    for name, target_nm in channel_order:
        band_index = nearest_band_index(wavelengths, target_nm)
        actual_nm = float(wavelengths[band_index])
        rgb_channels.append(
            normalize_channel(cube[..., band_index], lower, upper)
        )
        selected.append(
            {
                "channel": name,
                "target_wavelength_nm": float(target_nm),
                "actual_wavelength_nm": actual_nm,
                "band_index_0_based": band_index,
                "band_index_1_based": band_index + 1,
            }
        )

    return np.stack(rgb_channels, axis=-1), selected


def main():
    parser = argparse.ArgumentParser(
        description="Select real HSI bands to build pseudo-RGB images."
    )
    parser.add_argument("--input-root", default=str(ROOT))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "datasets" / "band_rgb_images"),
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "band_rgb_config.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dirs = discover_hsi_datasets(args.input_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No HSI dataset folders found in {args.input_root}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset_dir in dataset_dirs:
        loaded = load_envi(dataset_dir)
        cube = loaded["data"]
        wavelengths = parse_wavelengths(loaded["header"])
        rgb, selected = make_band_rgb(cube, wavelengths, config)

        png_path = output_dir / f"{dataset_dir.name}_bandrgb.png"
        Image.fromarray(rgb).save(png_path)
        records.append(
            {
                "dataset_id": dataset_dir.name,
                "input": str(dataset_dir),
                "png": str(png_path),
                "selected_bands": selected,
            }
        )

    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "config": config,
        "dataset_count": len(records),
        "saved": records,
    }
    metadata_path = output_dir / "band_rgb_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated {len(records)} band-selected pseudo-RGB images.")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
