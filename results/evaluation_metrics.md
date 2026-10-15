# RoadGuard AI — Evaluation Results

## Setup

- Model: YOLO11n, fine-tuned from pretrained weights
- Training duration: 30 epochs
- Image size: 640 × 640
- Hardware: NVIDIA Tesla T4 (Google Colab)
- Data split: 1,607 training images, 201 validation images, 201 held-out test images
- Dataset: [Road Damage Dataset: Potholes, Cracks and Manholes](https://zenodo.org/records/17834373)
- The dataset contains real-world photographs and annotated objects; no AI-generated training images were intentionally added by this project.

## Held-out test set

| Metric | Result |
|---|---:|
| Precision | 0.5403 (54.03%) |
| Recall | 0.5105 (51.05%) |
| mAP@50 | 0.5038 (50.38%) |
| mAP@50–95 | 0.2141 (21.41%) |

Test set: 201 images and 465 annotated instances.

## Per-class results

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Pothole | 0.470 | 0.450 | 0.428 | 0.165 |
| Crack | 0.438 | 0.363 | 0.342 | 0.128 |
| Manhole | 0.713 | 0.719 | 0.741 | 0.349 |

## Interpretation

The current model performs best on manholes. Crack detection is the weakest of the three classes, which suggests that further annotation review, class-aware error analysis, and additional training experiments may be useful. These are baseline results from one training run and should not be described as production-grade performance.

## Validation results after training

| Metric | Result |
|---|---:|
| Precision | 0.492 |
| Recall | 0.484 |
| mAP@50 | 0.479 |
| mAP@50–95 | 0.217 |

## Annotation preprocessing note

The source label files contained normalized four-corner polygons. They were converted to axis-aligned YOLO bounding boxes by taking the minimum and maximum x/y coordinates. Coordinates slightly outside the 0–1 range were clipped to image bounds. One zero-area annotation was skipped because its vertical coordinates were identical. This preprocessing should be clearly documented in any academic report.

## Limitations

- The test set is modest in size (201 images).
- Performance varies by damage category.
- Predictions may miss defects or produce false positives; no detection does not imply a road is safe.
- Dataset and model licenses should be reviewed before redistribution or commercial use.
