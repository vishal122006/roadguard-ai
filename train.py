"""Fine-tune YOLO11n on a YOLO-formatted road-damage dataset."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Initial pretrained weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU index like 0, or cpu")
    parser.add_argument("--project", default="runs/roadguard")
    parser.add_argument("--name", default="roadguard_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_period=5,
        seed=42,
        plots=True,
    )


if __name__ == "__main__":
    main()
