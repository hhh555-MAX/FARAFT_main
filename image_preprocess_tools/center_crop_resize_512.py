from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def center_crop_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def process_images(input_dir: Path, output_dir: Path, size: int) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_size = image.size
            image = center_crop_to_square(image)
            image = image.resize((size, size), Image.Resampling.BICUBIC)

            output_path = output_dir / image_path.name
            image.save(output_path)

        print(f"{image_path.name}: {original_size[0]}x{original_size[1]} -> {size}x{size}")

    print(f"Done. Saved {len(image_paths)} images to {output_dir}")


def main() -> None:
    parser = ArgumentParser(
        description="Center-crop images to the largest square, then resize to 512x512."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets"),
        help="Input image directory. Default: datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets_512_center_square"),
        help="Output image directory. Default: datasets_512_center_square",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output square size. Default: 512",
    )
    args = parser.parse_args()

    process_images(args.input, args.output, args.size)


if __name__ == "__main__":
    main()
