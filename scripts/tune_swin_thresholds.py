"""Tune class-specific confidence thresholds on validation; evaluate frozen thresholds on test.

Metric: greedy one-to-one box matching at IoU 0.50. Thresholds are operating points,
not calibrated probabilities. Requires compatible MMDetection runtime.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (aa + ab - inter) if aa + ab > inter else 0.0

def count_metrics(image_ids, ground_truth, predictions, class_id, threshold, iou_cut):
    tp = fp = fn = 0
    for image_id in image_ids:
        gts = ground_truth[image_id].get(class_id, [])
        preds = [p for p in predictions[image_id] if p["label"] == class_id and p["score"] >= threshold]
        preds.sort(key=lambda p: p["score"], reverse=True)
        matched = set()
        for p in preds:
            choices = [(iou(p["box"], gt), idx) for idx, gt in enumerate(gts) if idx not in matched]
            best_iou, best_idx = max(choices, default=(0.0, None))
            if best_idx is not None and best_iou >= iou_cut:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    precision = safe_div(tp, tp+fp)
    recall = safe_div(tp, tp+fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": safe_div(2*precision*recall, precision+recall)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/roadguard_swin_faster_rcnn.py")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-root", default=Path("data/RDD2022_coco"), type=Path)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--mode", choices=["tune", "evaluate"], default="tune")
    p.add_argument("--thresholds-json", type=Path, default=Path("checkpoints/best_thresholds.json"))
    p.add_argument("--out", type=Path, default=Path("runs/threshold_analysis"))
    p.add_argument("--iou", type=float, default=.50)
    p.add_argument("--step", type=float, default=.025)
    p.add_argument("--max-images", type=int, default=0, help="Debug only; 0 means all images.")
    a = p.parse_args()
    if a.mode == "tune" and a.split != "val":
        p.error("Threshold tuning must use --split val.")
    if a.mode == "evaluate" and a.split != "test":
        p.error("Frozen-threshold evaluation must use --split test.")
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    coco = json.loads((a.data_root / f"{a.split}.json").read_text(encoding="utf-8"))
    images = coco.get("images", [])
    if a.max_images:
        images = images[:a.max_images]
    image_ids = [int(x["id"]) for x in images]
    cat_names = {int(c["id"]): c["name"] for c in coco.get("categories", [])}
    cat_label = {cid: CLASSES.index(name) for cid, name in cat_names.items() if name in CLASSES}
    gt = defaultdict(lambda: defaultdict(list))
    for ann in coco.get("annotations", []):
        cid = int(ann["category_id"])
        if cid not in cat_label:
            continue
        x, y, w, h = map(float, ann["bbox"])
        if w > 0 and h > 0:
            gt[int(ann["image_id"])][cat_label[cid]].append([x, y, x+w, y+h])
    frozen = {}
    if a.mode == "evaluate":
        frozen = json.loads(a.thresholds_json.read_text(encoding="utf-8")).get("per_class_thresholds", {})
        if not all(name in frozen for name in CLASSES):
            raise ValueError("Threshold file must contain D00, D10, D20 and D40 thresholds.")
    import torch
    from mmdet.apis import init_detector, inference_detector
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(a.config, str(a.checkpoint), device=device)
    pred = {}
    root = a.data_root / "images"
    for i, row in enumerate(images, 1):
        path = root / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
        result = inference_detector(model, bgr).pred_instances
        boxes = result.bboxes.detach().cpu().numpy()
        scores = result.scores.detach().cpu().numpy()
        labels = result.labels.detach().cpu().numpy()
        pred[int(row["id"])] = [{"box": box.tolist(), "score": float(score), "label": int(label)}
                                for box, score, label in zip(boxes, scores, labels)
                                if 0 <= int(label) < len(CLASSES)]
        if i % 50 == 0 or i == len(images):
            print(f"Inferred {i}/{len(images)} {a.split} images")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "tune":
        thresholds = np.arange(a.step, 1.0, a.step).round(4).tolist()
        sweep = []
        for class_id, name in enumerate(CLASSES):
            for threshold in thresholds:
                row = {"class": name, "threshold": threshold,
                       **count_metrics(image_ids, gt, pred, class_id, threshold, a.iou)}
                sweep.append(row)
        csv_path = a.out / "validation_threshold_sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["class", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
            w.writeheader(); w.writerows(sweep)
        selected = {}
        selected_metrics = {}
        for name in CLASSES:
            rows = [r for r in sweep if r["class"] == name]
            best = max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"]))
            selected[name] = float(best["threshold"])
            selected_metrics[name] = best
        payload = {"dataset_split_used_for_selection": "val", "iou_threshold_for_f1": a.iou,
                   "selection_rule": "Per-class validation F1 maximum; ties prefer precision, then recall.",
                   "note": "Operating thresholds, not probability calibration. Freeze before test evaluation.",
                   "per_class_thresholds": selected, "validation_metrics_at_selected_threshold": selected_metrics}
        out_json = a.out / "best_thresholds.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved validation threshold sweep: {csv_path}")
        print(f"Saved frozen thresholds: {out_json}")
        print(json.dumps(payload, indent=2))
    else:
        rows = []
        for cid, name in enumerate(CLASSES):
            rows.append({"class": name, "threshold": float(frozen[name]),
                         **count_metrics(image_ids, gt, pred, cid, float(frozen[name]), a.iou)})
        tp = sum(x["tp"] for x in rows); fp = sum(x["fp"] for x in rows); fn = sum(x["fn"] for x in rows)
        precision = safe_div(tp, tp+fp); recall = safe_div(tp, tp+fn)
        rows.append({"class": "MICRO_AGGREGATE", "threshold": "per-class", "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": safe_div(2*precision*recall, precision+recall)})
        rows.append({"class": "MACRO_AVERAGE", "threshold": "per-class", "tp": "", "fp": "", "fn": "",
                     "precision": float(np.mean([r["precision"] for r in rows[:4]])),
                     "recall": float(np.mean([r["recall"] for r in rows[:4]])),
                     "f1": float(np.mean([r["f1"] for r in rows[:4]]))})
        path = a.out / "test_metrics_frozen_thresholds.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["class", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
            w.writeheader(); w.writerows(rows)
        print(f"Saved frozen-threshold test metrics: {path}")
        print(json.dumps(rows, indent=2))
        print("Report this F1 result separately from the COCO mAP metrics produced by tools/test.py.")

if __name__ == "__main__":
    main()
