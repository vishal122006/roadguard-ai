# RoadGuard Swin Transformer migration

## What's in this branch

The proposed replacement is **Swin-T + FPN + Faster R-CNN** using MMDetection. It targets four RDD2022 labels: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), and D40 (pothole). The original YOLO app and training workflow are left intact for rollback.

The multi-scale candidate uses multi-scale training sizes (640–800), smaller RPN anchor scales, AdamW, a lower learning-rate multiplier for pretrained backbone layers, gradient clipping, cosine decay, and validation checkpoint selection. A controlled fixed-640 reference config is included. These are experimental choices, not guaranteed improvements.

## 1. Dataset

Prepare labelled Pascal VOC XML data:

```text
data/RDD2022/images/       # original road photos
data/RDD2022/annotations/  # matching XML files
```

Only use labelled images for quantitative evaluation. Preserve the official test set if you already have a valid labelled split; do not overwrite it with random data. If the dataset contains adjacent video frames/routes, split by route or sequence before evaluating to avoid leakage.

Convert the XML dataset to COCO JSON with the migration package's converter or an equivalent vetted converter. The expected output is:

```text
data/RDD2022_coco/
  images/  train.json  val.json  test.json
```

The category IDs in JSON must map to D00, D10, D20, D40 in that order.

## 2. Environment and training

Use a GPU runtime (Google Colab is acceptable). MMDetection requires compatible PyTorch, CUDA, MMCV, MMEngine, and MMDetection versions; install from the official compatibility instructions rather than blindly upgrading an existing environment.

From an MMDetection source checkout, place the configs into its `configs/` folder and place the dataset at `data/RDD2022_coco/`. Then run:

```bash
python tools/train.py configs/roadguard_swin_faster_rcnn.py
python tools/test.py configs/roadguard_swin_faster_rcnn.py work_dirs/roadguard_swin/best_coco_bbox_mAP_epoch_*.pth
```

Choose the checkpoint using validation bbox mAP, then evaluate the held-out test split once. Confirm the exact checkpoint filename written under `work_dirs/roadguard_swin/`.

For a fair ablation, also train `configs/roadguard_swin_faster_rcnn_baseline.py` on the same split and seed. Compare identical test data and report the differences rather than assuming the tuned candidate is better.

## 3. Metrics to report

- COCO bbox mAP@0.50 and mAP@0.50:0.95, plus AP by class.
- Precision, recall, and F1 at a stated IoU (use 0.50 unless the report specifies another).
- Macro and micro averages, per-class false positives and false negatives.
- Inference latency/FPS, checkpoint size, parameter count, GPU and software versions.

Choose confidence thresholds on the validation set only, freeze them, then compute F1 on the held-out test set. Do not tune the threshold on test data. Confidence scores are not automatically calibrated probabilities.

## 4. Application

Train a checkpoint first and copy it to `checkpoints/roadguard_swin.pth`. Then run:

```bash
pip install streamlit pillow numpy
# Install the compatible MMDetection runtime as described above.
streamlit run app_swin.py
```

The new UI displays the original image next to the annotated output. It intentionally errors if the real Swin checkpoint is missing rather than silently running a different model or drawing demo boxes.

## 5. Accuracy improvement protocol

The prior YOLO values are historical results and must not be compared with a new run unless the split and metric protocol match. Keep the existing YOLO checkpoint and app for rollback until Swin training and validation pass. No performance scores are claimed by this branch; the dataset must be trained and measured first.
