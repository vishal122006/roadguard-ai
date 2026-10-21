# RoadGuard AI

**RoadGuard AI** is an academic computer-vision project for detecting visible road damage from road photographs using YOLO and a Streamlit interface.

> **Research prototype:** Predictions can be incorrect or incomplete and should not be used as the sole basis for road-safety, infrastructure, or maintenance decisions.

## Features

- Upload road photographs and visualize model-produced bounding boxes.
- Shows class names and confidence scores from the actual loaded checkpoint.
- Uses the project's own fine-tuned checkpoint when it is installed at `weights/best.pt` or configured through `ROADGUARD_MODEL_PATH`.
- If the project checkpoint is missing, uses a publicly hosted road-damage baseline from [vinothvikas1987/pothole-detection-yolov8](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8) so real model inference can still run.
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
