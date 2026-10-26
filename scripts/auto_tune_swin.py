"""Validation-driven hyperparameter search for RoadGuard Swin-T + FPN + Faster R-CNN.

Ranks candidate training settings on validation COCO bbox mAP, refines the winner,
and writes logs/JSON. The held-out test set is never used for model selection.
"""
from __future__ import annotations
import argparse, copy, json, math, re, subprocess, sys
from pathlib import Path
from typing import Any

TRIALS = [
    {"name":"balanced_multiscale","lr":1e-4,"weight_decay":0.05,"anchors":[4,8],
     "scales":[(640,640),(704,704),(768,768),(800,800)],"eval_scale":800,"batch":2},
    {"name":"small_crack_high_resolution","lr":5e-5,"weight_decay":0.05,"anchors":[2,4,8],
     "scales":[(704,704),(768,768),(832,832)],"eval_scale":832,"batch":1},
    {"name":"regularized_candidate","lr":5e-5,"weight_decay":0.10,"anchors":[8,16],
     "scales":[(640,640),(704,704),(768,768)],"eval_scale":768,"batch":2},
]

def set_scale(pipeline, kind, value):
    for transform in pipeline:
        if transform.get("type") == kind:
            if kind == "RandomChoiceResize": transform["scales"] = list(value)
            else: transform["scale"] = (value, value)
            return
    raise KeyError(f"Missing {kind} in pipeline {[t.get('type') for t in pipeline]}")

def make_config(base, out, work, trial, epochs, root, load_from=None, lr=None):
    from mmengine.config import Config
    cfg = Config.fromfile(str(base))
    root = root.resolve()
    cfg.data_root = str(root) + "/"
    for split in ("train", "val", "test"):
        loader = cfg[f"{split}_dataloader"]
        loader.dataset.data_root = str(root) + "/"
        loader.dataset.ann_file = f"{split}.json"
        loader.dataset.data_prefix = dict(img="images/")
    cfg.val_evaluator.ann_file = str(root / "val.json")
    cfg.test_evaluator.ann_file = str(root / "test.json")
    cfg.work_dir = str(work.resolve())
    cfg.load_from = str(load_from.resolve()) if load_from else None
    cfg.resume = False
    cfg.randomness.seed = 42
    cfg.train_cfg.max_epochs = int(epochs)
    cfg.train_cfg.val_interval = 1
    cfg.train_dataloader.batch_size = int(trial["batch"])
    cfg.train_dataloader.num_workers = 2
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1
    cfg.model.rpn_head.anchor_generator.scales = list(trial["anchors"])
    cfg.model.test_cfg.rcnn.score_thr = 0.001
    cfg.model.test_cfg.rcnn.max_per_img = 300
    cfg.optim_wrapper.optimizer.lr = float(lr if lr is not None else trial["lr"])
    cfg.optim_wrapper.optimizer.weight_decay = float(trial["weight_decay"])
    set_scale(cfg.train_dataloader.dataset.pipeline, "RandomChoiceResize", trial["scales"])
    set_scale(cfg.val_dataloader.dataset.pipeline, "Resize", trial["eval_scale"])
    set_scale(cfg.test_dataloader.dataset.pipeline, "Resize", trial["eval_scale"])
    n_images = len(json.loads((root / "train.json").read_text(encoding="utf-8")).get("images", []))
    warmup = min(500, max(20, math.ceil(n_images / trial["batch"]) * 2))
    for sch in cfg.param_scheduler:
        if sch.get("type") == "CosineAnnealingLR": sch.end = int(epochs)
        elif sch.get("type") == "LinearLR": sch.end = warmup
    cfg.default_hooks.checkpoint.interval = 1
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    cfg.default_hooks.checkpoint.save_best = "coco/bbox_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    out.parent.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(out))
    return cfg

def run_logged(command, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", " ".join(map(str, command)), flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Command failed ({proc.returncode}); log={log_path}\n{tail}")

def best_checkpoint(work):
    found = list(work.glob("best_coco_bbox_mAP_epoch_*.pth"))
    if found: return max(found, key=lambda p:p.stat().st_mtime)
    found = list(work.glob("epoch_*.pth"))
    if found:
        return max(found, key=lambda p:int(re.search(r"epoch_(\d+)",p.name).group(1)) if re.search(r"epoch_(\d+)",p.name) else -1)
    raise FileNotFoundError(f"No checkpoint in {work}")

def eval_validation(config, checkpoint, root, out_dir):
    """Run a real MMDetection evaluator on val.json, never test.json."""
    from mmengine.config import Config
    from mmengine.runner import Runner
    cfg = Config.fromfile(str(config))
    root = root.resolve()
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset.data_root = str(root) + "/"
    cfg.test_dataloader.dataset.ann_file = "val.json"
    cfg.test_dataloader.dataset.data_prefix = dict(img="images/")
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.test_evaluator.ann_file = str(root / "val.json")
    cfg.load_from = str(checkpoint.resolve()); cfg.resume = False
    cfg.work_dir = str(out_dir.resolve())
    cfg.train_cfg = None; cfg.train_dataloader = None
    cfg.val_cfg = None; cfg.val_dataloader = None; cfg.val_evaluator = None
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = Runner.from_cfg(cfg).test()
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("Validation evaluator returned no metrics; inspect MMDetection logs.")
    return {str(k): float(v) if isinstance(v,(int,float)) else str(v) for k,v in metrics.items()}

def get_map(metrics):
    for key in ("coco/bbox_mAP","bbox_mAP","coco/bbox_mAP_50"):
        if key in metrics: return float(metrics[key])
    for k,v in metrics.items():
        if "bbox_mAP" in k and "50" not in k and isinstance(v,(int,float)): return float(v)
    raise KeyError(f"No bbox mAP found in {metrics}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--mmdet-root",required=True,type=Path)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--search-epochs",type=int,default=8)
    p.add_argument("--final-epochs",type=int,default=24)
    p.add_argument("--skip-refinement",action="store_true")
    a=p.parse_args()
    a.config=a.config.resolve(); a.mmdet_root=a.mmdet_root.resolve()
    a.data_root=a.data_root.resolve(); a.out_dir=a.out_dir.resolve()
    for path in (a.config,a.mmdet_root,a.data_root/"train.json",a.data_root/"val.json",a.data_root/"test.json"):
        if not path.exists(): raise FileNotFoundError(path)
    if a.search_epochs < 2 or a.final_epochs < 1: p.error("search epochs must be >=2 and final epochs >=1")
    a.out_dir.mkdir(parents=True,exist_ok=True)
    summary={"selection_split":"val","selection_metric":"coco/bbox_mAP","search_epochs_each":a.search_epochs,
             "final_epochs":a.final_epochs,"seed":42,"trials":[],"note":"No score is guaranteed; test split is not used for selection."}
    successes=[]
    for trial in TRIALS:
        td=a.out_dir/"trials"/trial["name"]; cfg_path=a.out_dir/"configs"/f'{trial["name"]}.py'
        print(f"\n=== Trial {trial['name']} ===",flush=True)
        try:
            make_config(a.config,cfg_path,td,trial,a.search_epochs,a.data_root)
            run_logged([sys.executable,"tools/train.py",str(cfg_path)],a.mmdet_root,td/"train.log")
            ckpt=best_checkpoint(td)
            metrics=eval_validation(cfg_path,ckpt,a.data_root,td/"validation_eval")
            row={"name":trial["name"],"status":"success","parameters":trial,"config":str(cfg_path),
                 "work_dir":str(td),"checkpoint":str(ckpt),"validation_metrics":metrics,"selection_score_coco_bbox_mAP":get_map(metrics)}
            successes.append(row)
        except Exception as e:
            row={"name":trial["name"],"status":"failed","parameters":trial,"error":f"{type(e).__name__}: {e}"}
            print("Trial failed:",row["error"],flush=True)
        summary["trials"].append(row)
        (a.out_dir/"autotune_progress.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if not successes: raise RuntimeError(f"Every trial failed. See logs in {a.out_dir/'trials'}")
    winner=max(successes,key=lambda x:x["selection_score_coco_bbox_mAP"])
    summary["selected_search_trial"]=winner
    final_cfg=Path(winner["config"]); final_ckpt=Path(winner["checkpoint"]); final_work=Path(winner["work_dir"])
    final_metrics=winner["validation_metrics"]
    if not a.skip_refinement:
        final_work=a.out_dir/"final_refinement"
        final_cfg=a.out_dir/"configs"/"final_refinement.py"
        trial=winner["parameters"]
        make_config(Path(winner["config"]),final_cfg,final_work,trial,a.final_epochs,a.data_root,load_from=final_ckpt,lr=trial["lr"]*0.5)
        run_logged([sys.executable,"tools/train.py",str(final_cfg)],a.mmdet_root,final_work/"train.log")
        final_ckpt=best_checkpoint(final_work)
        final_metrics=eval_validation(final_cfg,final_ckpt,a.data_root,final_work/"validation_eval")
    summary.update({"final_config":str(final_cfg),"final_checkpoint":str(final_ckpt),"final_work_dir":str(final_work),
                    "final_validation_metrics":final_metrics,"final_validation_coco_bbox_mAP":get_map(final_metrics)})
    (a.out_dir/"autotune_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("AUTOTUNE COMPLETE:",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
