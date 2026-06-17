import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(CORE) not in sys.path:
    sys.path.append(str(CORE))

from config.parser import json_to_args  # noqa: E402
from gcsraft_certainty import GCSRAFT_certainty  # noqa: E402
from hraft import HomoRAFT  # noqa: E402
from raft import RAFT  # noqa: E402
from utils.flow_viz import flow_to_image  # noqa: E402
from utils.frame_utils import readFlow  # noqa: E402
from utils.utils import load_ckpt  # noqa: E402
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask  # noqa: E402
from utils_flow.pixel_wise_mapping import remap_using_flow_fields  # noqa: E402


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.array(image, dtype=np.float32).transpose(2, 0, 1)
    return torch.from_numpy(array).unsqueeze(0).to(device)


def build_model(args):
    if args.model == "RAFT":
        model = RAFT(args)
    elif args.model == "HomoRAFT":
        model = HomoRAFT(args)
    elif args.model == "GCSRAFT_certainty":
        model = GCSRAFT_certainty(args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    load_ckpt(model, args, distributed=False)
    return model


@torch.no_grad()
def predict_flow(model, args, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    output = model(image1, image2, iters=args.iters, test_mode=True)
    if "final" in output:
        return output["final"]
    return output["flow"][-1]


def compute_metrics(pred_flow: np.ndarray, gt_flow: np.ndarray) -> dict:
    valid = get_gt_correspondence_mask(gt_flow).astype(bool)
    epe_map = np.sqrt(np.sum((pred_flow - gt_flow) ** 2, axis=-1))
    valid_epe = epe_map[valid]

    if valid_epe.size == 0:
        return {
            "epe": float("nan"),
            "bad_1px": float("nan"),
            "bad_3px": float("nan"),
            "bad_5px": float("nan"),
            "valid_ratio": 0.0,
        }

    return {
        "epe": float(valid_epe.mean()),
        "bad_1px": float(100.0 * np.mean(valid_epe >= 1.0)),
        "bad_3px": float(100.0 * np.mean(valid_epe >= 3.0)),
        "bad_5px": float(100.0 * np.mean(valid_epe >= 5.0)),
        "valid_ratio": float(valid.mean()),
    }


def save_visuals(output_dir: Path, stem: str, image1_path: Path, image2_path: Path, pred_flow: np.ndarray, gt_flow: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image1 = np.array(Image.open(image1_path).convert("RGB"))
    image2 = np.array(Image.open(image2_path).convert("RGB"))

    pred_warp = remap_using_flow_fields(image2, pred_flow[..., 0], pred_flow[..., 1])
    gt_warp = remap_using_flow_fields(image2, gt_flow[..., 0], gt_flow[..., 1])

    cv2.imwrite(str(output_dir / f"{stem}_pred_flow.jpg"), flow_to_image(pred_flow, convert_to_bgr=True))
    cv2.imwrite(str(output_dir / f"{stem}_gt_flow.jpg"), flow_to_image(gt_flow, convert_to_bgr=True))
    cv2.imwrite(str(output_dir / f"{stem}_pred_warp.jpg"), cv2.cvtColor(pred_warp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{stem}_gt_warp.jpg"), cv2.cvtColor(gt_warp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{stem}_reference.jpg"), cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = ArgumentParser(description="Evaluate a trained model on synthetic image pairs with .flo ground truth.")
    parser.add_argument("--cfg", type=Path, default=Path("config/eval/dunhuang.json"))
    parser.add_argument("--model", required=True, choices=["RAFT", "HomoRAFT", "GCSRAFT_certainty"])
    parser.add_argument("--restore_ckpt", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/dunhuang/synthetic_512_all"))
    parser.add_argument("--output", type=Path, default=Path("demo/synthetic_512_eval"))
    parser.add_argument("--save_visuals", action="store_true")
    args_cli = parser.parse_args()

    args = json_to_args(args_cli.cfg)
    args.model = args_cli.model
    args.restore_ckpt = str(args_cli.restore_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This project expects CUDA for model inference.")

    model = build_model(args).to(device)
    model.eval()

    image_dir = args_cli.dataset / "images"
    flow_dir = args_cli.dataset / "flow"
    image1_paths = sorted(image_dir.glob("*_img_2.jpg"))
    if not image1_paths:
        raise FileNotFoundError(f"No *_img_2.jpg images found in {image_dir}")

    args_cli.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for image1_path in image1_paths:
        stem = image1_path.name.replace("_img_2.jpg", "")
        image2_path = image_dir / f"{stem}_img_1.jpg"
        flow_path = flow_dir / f"{stem}_flow.flo"

        if not image2_path.exists() or not flow_path.exists():
            print(f"Skip {stem}: missing image2 or flow")
            continue

        image1 = load_rgb_tensor(image1_path, device)
        image2 = load_rgb_tensor(image2_path, device)
        gt_flow = readFlow(str(flow_path)).astype(np.float32)

        pred = predict_flow(model, args, image1, image2)
        pred_flow = pred[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        metrics = compute_metrics(pred_flow, gt_flow)
        row = {"name": stem, **metrics}
        rows.append(row)
        print(
            f"{stem}: EPE={metrics['epe']:.3f}, "
            f"bad1={metrics['bad_1px']:.2f}%, bad3={metrics['bad_3px']:.2f}%, bad5={metrics['bad_5px']:.2f}%"
        )

        if args_cli.save_visuals:
            save_visuals(args_cli.output / "visuals", stem, image1_path, image2_path, pred_flow, gt_flow)

    if not rows:
        raise RuntimeError("No valid samples were evaluated.")

    csv_path = args_cli.output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "epe", "bad_1px", "bad_3px", "bad_5px", "valid_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    mean_epe = float(np.nanmean([row["epe"] for row in rows]))
    mean_bad1 = float(np.nanmean([row["bad_1px"] for row in rows]))
    mean_bad3 = float(np.nanmean([row["bad_3px"] for row in rows]))
    mean_bad5 = float(np.nanmean([row["bad_5px"] for row in rows]))
    print(f"Mean: EPE={mean_epe:.3f}, bad1={mean_bad1:.2f}%, bad3={mean_bad3:.2f}%, bad5={mean_bad5:.2f}%")
    print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
