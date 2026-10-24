"""Measure real batch-1 MMDetection inference latency and model size."""
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/roadguard_swin_faster_rcnn.py")
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path,help="One image or a folder of real road photos")
    p.add_argument("--max-images",type=int,default=50)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--out",type=Path,default=Path("runs/inference_benchmark.json"))
    a=p.parse_args()
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    if not a.source.exists(): raise FileNotFoundError(a.source)
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    paths=[a.source] if a.source.is_file() else sorted(x for x in a.source.rglob("*") if x.suffix.lower() in exts)
    paths=paths[:max(1,a.max_images)]
    if not paths: raise ValueError("No input images found")
    import torch
    from mmdet.apis import init_detector,inference_detector
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    model=init_detector(a.config,str(a.checkpoint),device=device); model.eval()
    arrays=[]
    for path in paths:
        with Image.open(path) as im: arrays.append(np.asarray(im.convert("RGB"))[:,:,::-1].copy())
    for x in arrays[:max(0,a.warmup)]: inference_detector(model,x)
    def sync():
        if device.startswith("cuda"): torch.cuda.synchronize()
    times=[]
    for x in arrays:
        sync(); start=time.perf_counter(); inference_detector(model,x); sync()
        times.append((time.perf_counter()-start)*1000)
    size=a.checkpoint.stat().st_size
    payload={"device":device,"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             "torch_version":torch.__version__,"cuda_version":torch.version.cuda,"images_measured":len(times),
             "latency_ms_mean":statistics.mean(times),"latency_ms_median":statistics.median(times),
             "latency_ms_p95":float(np.percentile(times,95)),
             "fps_batch1_from_mean_latency":1000/statistics.mean(times),
             "parameter_count":sum(p.numel() for p in model.parameters()),
             "checkpoint_size_bytes":size,"checkpoint_size_mib":size/(1024**2),
             "note":"Measured for this hardware, software, checkpoint and image set only."}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); print(f"Saved: {a.out.resolve()}")
if __name__=="__main__": main()
