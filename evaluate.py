"""Evaluate a trained YOLO checkpoint on a held-out dataset split."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--name", default="roadguard_test")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
