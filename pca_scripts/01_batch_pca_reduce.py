import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许脚本从任意工作目录启动时仍能导入项目模块。
    sys.path.insert(0, str(ROOT))

from pca_reduce_hsi import flatten_hsi, fit_pca, load_hsi, save_outputs, transform_pca
from utils.hsi_io import discover_hsi_datasets


def load_config(config_path):
    """读取可选 JSON 配置；命令行参数可以覆盖配置值。"""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sample_pixels(image, max_pixels, rng):
    """可选地采样像素，用于加速多张高光谱图像的 PCA 拟合。"""
    samples = flatten_hsi(image)
    if max_pixels <= 0 or samples.shape[0] <= max_pixels:
        return samples
    indices = rng.choice(samples.shape[0], size=max_pixels, replace=False)
    return samples[indices]


def fit_global_pca(dataset_dirs, components, standardize, max_fit_pixels, seed):
    """为所有数据集共同拟合一个共享 PCA 基底。"""
    rng = np.random.default_rng(seed)
    sampled = []
    shapes = {}
    for dataset_dir in dataset_dirs:
        image = load_hsi(dataset_dir)
        if image.shape[-1] != 204:
            raise ValueError(f"{dataset_dir} has {image.shape[-1]} bands, expected 204.")
        shapes[dataset_dir.name] = list(image.shape)
        sampled.append(sample_pixels(image, max_fit_pixels, rng))

    # 共享 PCA 基底可以保证不同图像的通道含义一致。
    fit_samples = np.concatenate(sampled, axis=0)
    return fit_pca(fit_samples, components, standardize), shapes


def reduce_with_global_pca(dataset_dirs, pca, output_dir):
    """把已经拟合好的 PCA 基底应用到每个数据集。"""
    records = []
    for dataset_dir in dataset_dirs:
        image = load_hsi(dataset_dir)
        reduced = transform_pca(image, pca)
        png_path, npy_path = save_outputs(reduced, output_dir, dataset_dir.name)
        records.append(
            {
                "dataset_id": dataset_dir.name,
                "input": str(dataset_dir),
                "png": str(png_path),
                "npy": str(npy_path),
            }
        )
    return records


def reduce_per_image(dataset_dirs, components, standardize, output_dir, max_fit_pixels, seed):
    """对每个数据集分别拟合并应用 PCA。"""
    rng = np.random.default_rng(seed)
    records = []
    explained = {}
    shapes = {}
    for dataset_dir in dataset_dirs:
        image = load_hsi(dataset_dir)
        if image.shape[-1] != 204:
            raise ValueError(f"{dataset_dir} has {image.shape[-1]} bands, expected 204.")
        shapes[dataset_dir.name] = list(image.shape)
        samples = sample_pixels(image, max_fit_pixels, rng)
        pca = fit_pca(samples, components, standardize)
        reduced = transform_pca(image, pca)
        png_path, npy_path = save_outputs(reduced, output_dir, dataset_dir.name)
        explained[dataset_dir.name] = pca["explained_variance_ratio"].tolist()
        records.append(
            {
                "dataset_id": dataset_dir.name,
                "input": str(dataset_dir),
                "png": str(png_path),
                "npy": str(npy_path),
            }
        )
    return records, explained, shapes


def main():
    parser = argparse.ArgumentParser(
        description="Batch-reduce Specim IQ HSI folders to PCA 3-channel images."
    )
    parser.add_argument("--input-root", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "datasets" / "pca_images"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "pca_config.json"))
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument(
        "--fit-mode",
        choices=["global", "per-image"],
        default=None,
        help="global fits one PCA basis for all datasets; per-image fits each dataset separately.",
    )
    parser.add_argument("--standardize", action="store_true", default=None)
    parser.add_argument(
        "--max-fit-pixels",
        type=int,
        default=0,
        help="Optional per-image pixel sample limit for PCA fitting. 0 means use all pixels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 配置文件提供稳定默认值；显式命令行参数优先级更高。
    config = load_config(args.config)
    components = args.components or int(config.get("components", 3))
    fit_mode = args.fit_mode or config.get("fit_mode", "global")
    standardize = (
        bool(config.get("standardize", False))
        if args.standardize is None
        else args.standardize
    )

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    dataset_dirs = discover_hsi_datasets(input_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No HSI dataset folders found in {input_root}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if fit_mode == "global":
        pca, shapes = fit_global_pca(
            dataset_dirs,
            components,
            standardize,
            args.max_fit_pixels,
            args.seed,
        )
        records = reduce_with_global_pca(dataset_dirs, pca, output_dir)
        explained = pca["explained_variance_ratio"].tolist()
    else:
        records, explained, shapes = reduce_per_image(
            dataset_dirs,
            components,
            standardize,
            output_dir,
            args.max_fit_pixels,
            args.seed,
        )

    # 元数据文件是后续仿射变换和 RAFT 步骤的交接文件。
    metadata = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "dataset_count": len(dataset_dirs),
        "fit_mode": fit_mode,
        "components": components,
        "standardize": standardize,
        "max_fit_pixels": args.max_fit_pixels,
        "input_shapes": shapes,
        "explained_variance_ratio": explained,
        "saved": records,
    }
    metadata_path = output_dir / "pca_batch_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Reduced {len(records)} HSI datasets with {fit_mode} PCA.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
