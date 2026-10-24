"""Convert labelled Pascal VOC XML road-damage data to COCO JSON.

Outputs train.json, val.json, test.json, images/, and a split manifest. Supports
recursive folders and an optional regex group key (e.g. route/video ID) to keep
related frames in the same split. Inspect the generated counts before training.
"""
import argparse
import csv
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]
ALIASES = {
    "d00": 0, "longitudinal crack": 0, "longitudinal": 0,
    "d10": 1, "transverse crack": 1, "transverse": 1,
    "d20": 2, "alligator crack": 2, "alligator": 2,
    "d40": 3, "pothole": 3,
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_class(name):
    key = " ".join((name or "").strip().lower().replace("_", " ").split())
    m = re.match(r"^(d00|d10|d20|d40)\b", key)
    if m:
        return {"d00": 0, "d10": 1, "d20": 2, "d40": 3}[m.group(1)]
    return ALIASES.get(key)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--out", default=Path("data/RDD2022_coco"), type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=.70)
    p.add_argument("--val-ratio", type=float, default=.15)
    p.add_argument("--group-regex", default="", help="Regex on relative image path; capture group 1 identifies related frames.")
    a = p.parse_args()
    if not a.images.is_dir() or not a.annotations.is_dir():
        p.error("--images and --annotations must be existing directories")
    if a.train_ratio <= 0 or a.val_ratio <= 0 or a.train_ratio + a.val_ratio >= 1:
        p.error("Ratios must be positive and train + val < 1")
    group_re = re.compile(a.group_regex) if a.group_regex else None
    imgs = sorted(x for x in a.images.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)
    by_rel = {x.relative_to(a.images).with_suffix("").as_posix().casefold(): x for x in imgs}
    by_stem = defaultdict(list)
    for x in imgs:
        by_stem[x.stem.casefold()].append(x)
    pairs, unmatched = [], []
    for xml in sorted(a.annotations.rglob("*.xml")):
        rel = xml.relative_to(a.annotations).with_suffix("").as_posix().casefold()
        image = by_rel.get(rel)
        if image is None:
            matches = by_stem.get(xml.stem.casefold(), [])
            if len(matches) == 1:
                image = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous image for {xml}; preserve matching relative folders.")
        if image is None:
            unmatched.append(str(xml))
        else:
            pairs.append((image, xml))
    if not pairs:
        raise SystemExit("No matching image/XML pairs found.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} XML files had no matching image; first 5: {unmatched[:5]}")
    rng = random.Random(a.seed)
    # If a sequence/group regex is supplied, related frames never cross partitions.
    grouped = defaultdict(list)
    group_by_image = {}
    for pair in pairs:
        image = pair[0]
        rel = image.relative_to(a.images).as_posix()
        match = group_re.search(rel) if group_re else None
        key = (match.group(1) if match and match.lastindex else match.group(0)) if match else rel
        grouped[key].append(pair)
        group_by_image[rel] = key
    groups = list(grouped.items())
    rng.shuffle(groups)
    n = len(pairs)
    limits = {"train": int(n * a.train_ratio), "val": int(n * a.val_ratio)}
    assignments = {"train": [], "val": [], "test": []}
    # Greedy fill preserves group boundaries and approximately respects requested ratios.
    targets = {"train": n * a.train_ratio, "val": n * a.val_ratio, "test": n * (1-a.train_ratio-a.val_ratio)}
    sizes = {k: 0 for k in assignments}
    for _, members in sorted(groups, key=lambda x: len(x[1]), reverse=True):
        chosen = min(assignments, key=lambda s: ((sizes[s] + len(members) - targets[s]) ** 2 / max(targets[s], 1), sizes[s]))
        assignments[chosen].extend(members)
        sizes[chosen] += len(members)
    for split in assignments:
        rng.shuffle(assignments[split])
    out_images = a.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    categories = [{"id": i+1, "name": name, "supercategory": "road_damage"} for i, name in enumerate(CLASSES)]
    manifest = []
    excluded = Counter()
    dimension_warnings = 0
    for split, split_pairs in assignments.items():
        coco_images, coco_anns = [], []
        ann_id = 1
        for image_id, (image_path, xml_path) in enumerate(split_pairs, start=1):
            with Image.open(image_path) as im:
                width, height = im.size
                im.verify()
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            old_w = int(size.findtext("width", width)) if size is not None else width
            old_h = int(size.findtext("height", height)) if size is not None else height
            sx, sy = width / max(old_w, 1), height / max(old_h, 1)
            dimension_warnings += int(old_w != width or old_h != height)
            rel = image_path.relative_to(a.images).as_posix()
            coco_images.append({"id": image_id, "file_name": rel, "width": width, "height": height})
            target = out_images / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != image_path.stat().st_size:
                shutil.copy2(image_path, target)
            labels_in_image, box_count = set(), 0
            for obj in root.findall("object"):
                name = obj.findtext("name", "")
                cid = parse_class(name)
                if cid is None:
                    excluded[name or "<missing>"] += 1
                    continue
                box = obj.find("bndbox")
                if box is None:
                    continue
                # Pascal VOC is 1-based/inclusive; COCO uses [x, y, width, height].
                x1 = max(0.0, (float(box.findtext("xmin", "0")) - 1) * sx)
                y1 = max(0.0, (float(box.findtext("ymin", "0")) - 1) * sy)
                x2 = min(float(width), float(box.findtext("xmax", "0")) * sx)
                y2 = min(float(height), float(box.findtext("ymax", "0")) * sy)
                if x2 <= x1 or y2 <= y1:
                    continue
                bw, bh = x2-x1, y2-y1
                coco_anns.append({"id": ann_id, "image_id": image_id, "category_id": cid+1,
                                  "bbox": [round(x1, 3), round(y1, 3), round(bw, 3), round(bh, 3)],
                                  "area": round(bw*bh, 3), "iscrowd": 0})
                ann_id += 1
                box_count += 1
                labels_in_image.add(CLASSES[cid])
            manifest.append({"split": split, "image": rel, "annotation": str(xml_path),
                             "group": group_by_image[rel], "classes_present": "|".join(sorted(labels_in_image)),
                             "num_boxes": box_count})
        payload = {"info": {"description": f"RoadGuard RDD2022 {split}", "version": "1.0"},
                   "licenses": [], "images": coco_images, "annotations": coco_anns, "categories": categories}
        (a.out / f"{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        counts = Counter(CLASSES[ann["category_id"]-1] for ann in coco_anns)
        print(f"{split}: images={len(coco_images)}, boxes={len(coco_anns)}, per-class boxes={dict(counts)}")
    with (a.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "annotation", "group", "classes_present", "num_boxes"])
        w.writeheader()
        w.writerows(manifest)
    summary = {"paired_images": len(pairs), "split_sizes": sizes, "classes": CLASSES,
               "excluded_non_target_labels": dict(excluded), "xml_image_dimension_mismatches_scaled": dimension_warnings,
               "seed": a.seed, "group_regex": a.group_regex or None}
    (a.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote COCO dataset to {a.out.resolve()}")
    print("Inspect sampled bounding-box overlays and split class counts before training.")
    if not group_re:
        print("NOTE: no group regex supplied; adjacent/duplicate frames may leak across splits if present.")

if __name__ == "__main__":
    main()
