import json
import math
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.interpolate import griddata


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FLOW_TAG = np.array([202021.25], np.float32)


def write_flo(path: Path, flow: np.ndarray) -> None:
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Flow must have shape HxWx2, got {flow.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = flow.shape[:2]
    with path.open("wb") as f:
        f.write(FLOW_TAG.tobytes())
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def coords_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return x, y


def homography_flow(height: int, width: int, H: np.ndarray) -> np.ndarray:
    x, y = coords_grid(height, width)
    ones = np.ones_like(x)
    points = np.stack([x, y, ones], axis=0).reshape(3, -1)
    warped = H @ points
    warped = warped[:2] / (warped[2:3] + 1e-8)
    warped_x = warped[0].reshape(height, width)
    warped_y = warped[1].reshape(height, width)
    return np.stack([warped_x - x, warped_y - y], axis=-1).astype(np.float32)


def random_affine_homography(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    angle = float(rng.uniform(-18.0, 18.0))
    scale = float(rng.uniform(0.86, 1.14))
    shear_x = math.radians(float(rng.uniform(-8.0, 8.0)))
    shear_y = math.radians(float(rng.uniform(-8.0, 8.0)))
    tx = float(rng.uniform(-0.08, 0.08) * size)
    ty = float(rng.uniform(-0.08, 0.08) * size)

    center = np.array([[1, 0, -size / 2], [0, 1, -size / 2], [0, 0, 1]], np.float32)
    uncenter = np.array([[1, 0, size / 2], [0, 1, size / 2], [0, 0, 1]], np.float32)
    rotation = np.array(
        [
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
            [0, 0, 1],
        ],
        np.float32,
    )
    scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    shear = np.array([[1, math.tan(shear_x), 0], [math.tan(shear_y), 1, 0], [0, 0, 1]], np.float32)
    translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float32)
    H = translate @ uncenter @ shear @ scale_mat @ rotation @ center
    meta = {
        "angle_deg": angle,
        "scale": scale,
        "shear_x_deg": math.degrees(shear_x),
        "shear_y_deg": math.degrees(shear_y),
        "tx_px": tx,
        "ty_px": ty,
    }
    return H.astype(np.float32), meta


def random_perspective_homography(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    margin = 0.10 * size
    src = np.array(
        [[0, 0], [size - 1, 0], [0, size - 1], [size - 1, size - 1]],
        np.float32,
    )
    dst = src + rng.uniform(-margin, margin, src.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    meta = {"src_corners": src.tolist(), "dst_corners": dst.tolist(), "margin_px": margin}
    return H.astype(np.float32), meta


def smooth_noise(rng: np.random.Generator, size: int, scale: float, sigma: float) -> np.ndarray:
    noise = rng.standard_normal((size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    max_abs = float(np.max(np.abs(noise)))
    if max_abs > 1e-6:
        noise = noise / max_abs
    return noise * scale


def random_elastic_flow(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    scale = float(rng.uniform(10.0, 28.0))
    sigma = float(rng.uniform(18.0, 42.0))
    dx = smooth_noise(rng, size, scale, sigma)
    dy = smooth_noise(rng, size, scale, sigma)
    flow = np.stack([dx, dy], axis=-1).astype(np.float32)
    return flow, {"scale_px": scale, "sigma": sigma}


def random_tps_like_flow(rng: np.random.Generator, size: int) -> tuple[np.ndarray, dict]:
    grid_size = 4
    strength = float(rng.uniform(10.0, 30.0))
    control_axis = np.linspace(0, size - 1, grid_size, dtype=np.float32)
    cx, cy = np.meshgrid(control_axis, control_axis)
    control_points = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=-1)
    offsets = rng.uniform(-strength, strength, control_points.shape).astype(np.float32)

    x, y = coords_grid(size, size)
    dx = griddata(control_points, offsets[:, 0], (x, y), method="cubic", fill_value=0.0)
    dy = griddata(control_points, offsets[:, 1], (x, y), method="cubic", fill_value=0.0)
    flow = np.stack([dx, dy], axis=-1).astype(np.float32)
    meta = {
        "grid_size": grid_size,
        "strength_px": strength,
        "control_points": control_points.tolist(),
        "offsets": offsets.tolist(),
    }
    return flow, meta


def warp_from_forward_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Create transformed image I2 so remap(I2, flow) approximately reconstructs I1."""
    height, width = flow.shape[:2]
    x, y = coords_grid(height, width)
    forward_x = x + flow[..., 0]
    forward_y = y + flow[..., 1]

    points = np.stack([forward_x.reshape(-1), forward_y.reshape(-1)], axis=-1)
    values_x = x.reshape(-1)
    values_y = y.reshape(-1)

    grid_points = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    map_x = griddata(points, values_x, grid_points, method="linear", fill_value=-1).reshape(height, width)
    map_y = griddata(points, values_y, grid_points, method="linear", fill_value=-1).reshape(height, width)
    map_x = np.nan_to_num(map_x, nan=-1.0, posinf=-1.0, neginf=-1.0)
    map_y = np.nan_to_num(map_y, nan=-1.0, posinf=-1.0, neginf=-1.0)

    return cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def generate_transform(rng: np.random.Generator, transform: str, size: int) -> tuple[np.ndarray, dict]:
    if transform == "affine":
        H, meta = random_affine_homography(rng, size)
        return homography_flow(size, size, H), {"type": transform, "H": H.tolist(), **meta}
    if transform == "perspective":
        H, meta = random_perspective_homography(rng, size)
        return homography_flow(size, size, H), {"type": transform, "H": H.tolist(), **meta}
    if transform == "tps":
        flow, meta = random_tps_like_flow(rng, size)
        return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0), {"type": transform, **meta}
    if transform == "elastic":
        flow, meta = random_elastic_flow(rng, size)
        return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0), {"type": transform, **meta}
    raise ValueError(f"Unsupported transform: {transform}")


def main() -> None:
    parser = ArgumentParser(description="Generate synthetic 512x512 image pairs and ground-truth .flo files.")
    parser.add_argument("--input", type=Path, default=Path("datasets_512_center_square"))
    parser.add_argument("--output", type=Path, default=Path("datasets/dunhuang/synthetic_512_all"))
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["affine", "perspective", "tps", "elastic"],
        choices=["affine", "perspective", "tps", "elastic"],
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    image_dir = args.output / "images"
    flow_dir = args.output / "flow"
    meta_dir = args.output / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    rng = np.random.default_rng(args.seed)
    sample_index = 0

    for image_path in image_paths:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = np.array(image)
        if image.shape[0] != args.size or image.shape[1] != args.size:
            image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)

        for transform in args.transforms:
            flow, metadata = generate_transform(rng, transform, args.size)
            transformed = warp_from_forward_flow(image, flow)

            stem = f"image_{sample_index:04d}"
            cv2.imwrite(str(image_dir / f"{stem}_img_2.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(image_dir / f"{stem}_img_1.jpg"), cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
            write_flo(flow_dir / f"{stem}_flow.flo", flow)

            metadata.update(
                {
                    "source_file": image_path.name,
                    "reference_image": f"{stem}_img_2.jpg",
                    "transformed_image": f"{stem}_img_1.jpg",
                    "flow_file": f"{stem}_flow.flo",
                    "flow_direction": "reference img_2 pixel -> transformed img_1 sampling coordinate",
                }
            )
            with (meta_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"{stem}: {image_path.name}, {transform}")
            sample_index += 1

    print(f"Done. Generated {sample_index} pairs in {args.output}")


if __name__ == "__main__":
    main()
