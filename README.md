# RoadGuard AI

**RoadGuard AI** is an academic computer-vision project that detects visible potholes, cracks, and manholes in road photographs using a fine-tuned YOLO11n object-detection model. It includes a Streamlit application for interactive predictions, training/evaluation scripts, and documented test-set results.

> **Research prototype:** Predictions can be incorrect or incomplete and should not be used as the sole basis for road-safety, infrastructure, or maintenance decisions.

## Features

- Detects three classes: `Pothole`, `Crack`, and `Manhole`.
- Fine-tunes pretrained YOLO11n weights.
- Reports precision, recall, mAP@50, and mAP@50–95.
- Provides a Streamlit interface for image upload, confidence-threshold selection, bounding-box visualization, and confidence scores.
- Uses real-world road-damage photographs; no AI-generated training images were intentionally added.

## Dataset

Dataset record: [Road Damage Dataset: Potholes, Cracks and Manholes (Zenodo)](https://zenodo.org/records/17834373)

The working dataset contained 2,009 photographs split into training (1,607), validation (201), and test (201) sets. The test split was held out from training and validation.

The original labels were polygon-style annotations. They were converted to axis-aligned YOLO boxes by clipping coordinates to image boundaries. One zero-area annotation was excluded because it had no height. The dataset is not included in this repository. Review the dataset record's license/terms before redistribution or commercial use.

## Test-set results

Results from the reported 30-epoch YOLO11n run on the held-out test set:

| Metric | Result |
|---|---:|
| Precision | 54.03% |
| Recall | 51.05% |
| mAP@50 | 50.38% |
| mAP@50–95 | 21.41% |

Per-class results and limitations are in [results/evaluation_metrics.md](results/evaluation_metrics.md). Manhole detection performed best in this run; crack detection needs improvement.

## Repository structure

```text
roadguard-ai/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── README.md
└── results/
    └── evaluation_metrics.md
```

## Setup

Python 3.10+ is recommended. A CUDA-capable NVIDIA GPU is recommended for training; CPU inference may work but will be slower.

```bash
git clone https://github.com/vishal122006/roadguard-ai.git
cd roadguard-ai
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

## Obtain the trained model checkpoint

The trained weights (`best.pt`) are not included in this source-only repository. Put your checkpoint at `weights/best.pt`, or set `ROADGUARD_MODEL_PATH` to its path. Do not commit large model files, dataset archives, secrets, or private data by accident.

Example (macOS/Linux):

```bash
export ROADGUARD_MODEL_PATH="/path/to/best.pt"
```

Windows PowerShell:

```powershell
$env:ROADGUARD_MODEL_PATH="C:\\path\\to\\best.pt"
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Upload a `.jpg`, `.jpeg`, or `.png` road image, choose a confidence threshold, and click **Detect road damage**. The app displays the original image, annotated predictions, class names, and confidence scores.

## Fine-tune the model

Prepare a YOLO-format dataset and a `data.yaml` with correct paths and class names.

```bash
python train.py --data /path/to/data.yaml --model yolo11n.pt --epochs 30 --imgsz 640 --batch 8 --device 0
```

## Evaluate on the held-out test split

```bash
python evaluate.py --model /path/to/best.pt --data /path/to/data.yaml --split test --device 0
```

Use validation data for model tuning and keep the test set for final evaluation.

## Attribution and academic transparency

- Dataset: [Road Damage Dataset: Potholes, Cracks and Manholes](https://zenodo.org/records/17834373). Consult the record for official citation and license/terms.
- Framework: [Ultralytics](https://github.com/ultralytics/ultralytics). Review applicable software and pretrained-weight terms before redistribution or commercial use.

This README describes one reported training run. Recalculate and update metrics if you train a new model. For academic submission, acknowledge AI assistance in accordance with your subject's rules and document your own implementation, experiments, and contributions.
