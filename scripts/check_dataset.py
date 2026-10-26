"""Validate COCO-format RoadGuard splits and referenced images before training."""
import argparse, json
from pathlib import Path
from collections import Counter
from PIL import Image

EXPECTED = ["D00","D10","D20","D40"]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--sample-images",type=int,default=20)
    a=p.parse_args()
    root=a.root.resolve()
    required=["train.json","val.json","test.json","images"]
    missing=[x for x in required if not (root/x).exists()]
    if missing: raise FileNotFoundError(f"Missing dataset items: {missing}")
    summary={"root":str(root),"splits":{},"expected_classes":EXPECTED,"warnings":[]}
    for split in ("train","val","test"):
        obj=json.loads((root/f"{split}.json").read_text(encoding="utf-8"))
        imgs=obj.get("images",[]); anns=obj.get("annotations",[]); cats=obj.get("categories",[])
        cat_names={int(c["id"]):c["name"] for c in cats}
        if not imgs: raise ValueError(f"{split} split has no images")
        if [cat_names.get(i) for i in sorted(cat_names)] != EXPECTED:
            raise ValueError(f"{split} categories should map IDs 1..4 to {EXPECTED}, found {cat_names}")
        image_ids={int(x["id"]) for x in imgs}
        per_class=Counter()
        for ann in anns:
            if int(ann["image_id"]) not in image_ids: raise ValueError(f"Annotation references missing image_id: {ann}")
            cid=int(ann["category_id"])
            if cid not in cat_names: raise ValueError(f"Unknown category id: {ann}")
            x,y,w,h=map(float,ann["bbox"])
            if w<=0 or h<=0: raise ValueError(f"Invalid box: {ann}")
            per_class[cat_names[cid]]+=1
        for im in imgs[:max(1,a.sample_images)]:
            path=root/"images"/im["file_name"]
            if not path.is_file(): raise FileNotFoundError(f"Image referenced by JSON is missing: {path}")
            with Image.open(path) as picture:
                if picture.size != (int(im["width"]),int(im["height"])):
                    raise ValueError(f"Image dimensions mismatch for {path}: {picture.size} vs JSON {(im['width'],im['height'])}")
        summary["splits"][split]={"images":len(imgs),"boxes":len(anns),"class_box_counts":dict(per_class)}
        if not anns: summary["warnings"].append(f"{split} split contains no annotations")
    print(json.dumps(summary,indent=2))
    (root/"dataset_check_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
if __name__=="__main__": main()
