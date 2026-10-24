# RoadGuard AI

**RoadGuard AI** is an academic computer-vision project for detecting visible road damage from road photographs using YOLO and a Streamlit interface.

> **Research prototype:** Predictions can be incorrect or incomplete and should not be used as the sole basis for road-safety, infrastructure, or maintenance decisions.

## Features

- Upload road photographs and visualize model-produced bounding boxes.
- Shows class names and confidence scores from the actual loaded checkpoint.
- Uses the project's own fine-tuned checkpoint when it is installed at `weights/best.pt` or configured through `ROADGUARD_MODEL_PATH`.
- If the project checkpoint is missing, uses a publicly hosted road-damage baseline from [vinothvikas1987/pothole-detection-yolov8](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8) so real model inference can still run.
- For detections labeled `Other`, optionally applies a second-stage TinyCLIP zero-shot classifier to the cropped image region and suggests the closest visual category (pothole, crack, manhole cover, patch, oil/fluid stain, road marking, debris, water, broken asphalt, or normal road surface).
- Shows the detector's original class and score separately from the second-stage visual match score. The visual match score is relative to the candidate list, not a calibrated probability. This is a general zero-shot estimate, not a replacement for training and validating a dedicated classifier on correctly labeled road-damage crops.
- No synthetic or fabricated detection results are generated.

## Checkpoint behavior

The project repository does not include its own `best.pt` file; weight files are intentionally excluded from Git. The app now uses this selection order:

1. If `ROADGUARD_MODEL_PATH` is set, load that exact file. If it is missing, the app reports an error instead of silently using a different model.
2. Otherwise, if `weights/best.pt` exists in the repository, load it.
3. Otherwise, download the public baseline checkpoint `best.pt` from Hugging Face and use that model.

The public fallback is a **third-party baseline, not the project's own trained checkpoint**. Its classes are Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole, and Other. The `Other` class may include manholes, patches, oil spills, and road markings; it is **not a dedicated manhole class**. Model details and its published Apache-2.0 license are available at the Hugging Face model page. Review all upstream model and dataset terms before redistribution or commercial use.

To disable automatic public fallback in Streamlit Cloud, set `ROADGUARD_USE_PUBLIC_BASELINE=false`. To specify your own trained model, add it as `weights/best.pt` or configure `ROADGUARD_MODEL_PATH`.

## Reported custom-model evaluation

The table below describes one previously reported YOLO11n run trained by this project; it does **not** describe the third-party fallback model.

| Metric | Project-reported test result |
|---|---:|
| Precision | 54.03% |
| Recall | 51.05% |
| mAP@50 | 50.38% |
| mAP@50–95 | 21.41% |

These values are documented historical results, not a guarantee of current deployment performance. Re-evaluate any new checkpoint on the held-out test set.

## Setup

Python 3.10+ is recommended.

```bash
git clone https://github.com/vishal122006/roadguard-ai.git
cd roadguard-ai
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first inference with the public fallback requires network access so Hugging Face can download the checkpoint. Subsequent loads use the Hugging Face local cache.

## Train and use your own model

Prepare a YOLO-formatted dataset with a `data.yaml` containing the correct class names and train/validation/test paths.

```bash
python train.py --data /path/to/data.yaml --model yolo11n.pt --epochs 30 --imgsz 640 --batch 8 --device 0
```

After training, copy the chosen checkpoint to `weights/best.pt` or configure `ROADGUARD_MODEL_PATH`. Your checkpoint must have class IDs and meanings consistent with the labels used during training. Re-evaluate the checkpoint on the held-out test set before reporting its metrics.

```bash
python evaluate.py --model /path/to/best.pt --data /path/to/data.yaml --split test --device 0
```

## Attribution

- Public fallback baseline: [vinothvikas1987/pothole-detection-yolov8](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8).
- Training/inference framework: [Ultralytics](https://github.com/ultralytics/ultralytics).

This is an academic prototype. For academic submission, document the exact model used for each result, cite upstream work, and follow your institution's AI-assistance policy.


## Second-stage classification of `Other` detections

When the public detector predicts `Other`, the app crops that region and runs [TinyCLIP ViT-8M](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) over a fixed set of candidate road-surface descriptions. It displays the highest-matching candidate as `Likely ...` or `Possible ... (weak visual match)`, along with the detector score and relative visual match score. The top alternatives are available in an expander.

TinyCLIP is a zero-shot vision-language classifier; its scores are relative across the supplied labels and are **not calibrated probabilities**. It may confuse visually similar categories. If it cannot load, the app explicitly reports that the type could not be estimated rather than fabricating a class. For robust domain accuracy, collect representative real crops with verified labels and train/evaluate a dedicated classification model.


---

## Swin Transformer edition (MAAI 901)

A migration candidate is available on the `feat/swin-transformer-roadguard` branch. It adds **Swin-T + FPN + Faster R-CNN** as an alternative to the existing YOLO pipeline, without silently switching the deployed app or overwriting the current checkpoint.

- Model config: `configs/roadguard_swin_faster_rcnn.py`
- Controlled reference config: `configs/roadguard_swin_faster_rcnn_baseline.py`
- Inference UI (requires a trained checkpoint): `app_swin.py`
- Pascal VOC conversion: `scripts/convert_voc_to_coco_swin.py`
- Validation-only class threshold tuning / frozen test F1: `scripts/tune_swin_thresholds.py`
- Latency and model-size measurements: `scripts/benchmark_swin.py`
- Full workflow: `SWIN_MIGRATION.md`

### Swin workflow

First prepare paired labelled XML and image files. Convert and audit them:

```bash
python scripts/convert_voc_to_coco_swin.py --images data/RDD2022/images --annotations data/RDD2022/annotations --out data/RDD2022_coco
```

Use an optional `--group-regex` when filenames encode route/video sequences so adjacent frames stay together. Inspect split class counts and sampled bounding-box overlays. For a dataset that already has an official labelled split, preserve it instead of creating a new random split.

Install a PyTorch/CUDA/MMCV/MMEngine/MMDetection combination that matches the official OpenMMLab compatibility matrix. From an MMDetection checkout, copy the configs into its `configs/` directory and put `data/RDD2022_coco/` under that checkout. Train:

```bash
python tools/train.py configs/roadguard_swin_faster_rcnn.py
python tools/test.py configs/roadguard_swin_faster_rcnn.py /path/to/selected_checkpoint.pth
```

Tune class-specific F1 thresholds on validation only, then freeze them and evaluate on test:

```bash
python scripts/tune_swin_thresholds.py --config configs/roadguard_swin_faster_rcnn.py --checkpoint /path/to/selected_checkpoint.pth --data-root data/RDD2022_coco --split val --mode tune --out runs/thresholds
python scripts/tune_swin_thresholds.py --config configs/roadguard_swin_faster_rcnn.py --checkpoint /path/to/selected_checkpoint.pth --data-root data/RDD2022_coco --split test --mode evaluate --thresholds-json runs/thresholds/best_thresholds.json --out runs/final_test
python scripts/benchmark_swin.py --config configs/roadguard_swin_faster_rcnn.py --checkpoint /path/to/selected_checkpoint.pth --source /path/to/real_road_photos --out runs/inference_benchmark.json
```

Copy the trained checkpoint to `checkpoints/roadguard_swin.pth` and run `streamlit run app_swin.py`. Set `ROADGUARD_CONFIG`, `ROADGUARD_CHECKPOINT`, and `ROADGUARD_THRESHOLDS` if you use different paths.

### Do not claim an improvement before measurement

The Swin config tests possible improvements through multi-scale training, smaller RPN anchors, pretrained Swin features, and conservative fine-tuning. It does **not** guarantee better precision, recall, F1, or mAP. Select model weights on validation mAP, select F1 thresholds on validation, then use the held-out test split once. Report mAP@0.50 and mAP@0.50:0.95, classwise AP, precision/recall/F1 at a stated IoU, per-class false positives/negatives, latency, hardware and software versions. The existing YOLO app remains available as a rollback until the trained Swin model has been reviewed.
