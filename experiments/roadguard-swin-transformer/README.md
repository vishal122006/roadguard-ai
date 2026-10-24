# RoadGuard AI — Swin Transformer experiment results

## Experiment configuration
- Model: Swin Transformer Tiny + FPN + Faster R-CNN
- Dataset: RDD2022, 4 classes
- Best selected checkpoint: `best_coco_bbox_mAP_epoch_5.pth`
- Training epochs completed: 5
- Confidence threshold: 0.50, selected on validation data
- Matching IoU threshold: 0.50
- Evaluation implementation: greedy one-to-one matching per image and class

## Validation COCO metrics at epoch 5
| Metric | Result |
|---|---:|
| mAP@0.50:0.95 | 9.00% |
| mAP@0.50 | 24.70% |
| mAP@0.75 | 4.00% |

## Validation confidence-threshold sweep
| Confidence threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 12.73% | 64.29% | 21.25% |
| 0.30 | 18.73% | 57.17% | 28.22% |
| 0.40 | 25.04% | 49.67% | 33.29% |
| 0.50 | 31.59% | 42.55% | 36.26% |

## Final test-set results
| Metric | Result |
|---|---:|
| Images processed | 1,156 |
| TP | 426 |
| FP | 949 |
| FN | 571 |
| Precision | 30.98% |
| Recall | 42.73% |
| F1 | 35.92% |

## Per-class test performance
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| D00_Longitudinal_Crack | 31.25% | 17.24% | 22.22% |
| D10_Transverse_Crack | 0.00% | 0.00% | 0.00% |
| D20_Alligator_Crack | 38.67% | 60.57% | 47.21% |
| D40_Pothole | 26.79% | 45.88% | 33.83% |

## Limitations and reporting notes
- The confidence threshold was selected by maximizing micro-F1 on the validation split among 0.20, 0.30, 0.40, and 0.50.
- An epoch-1 test evaluation was previously conducted at threshold 0.30, so the test split has been explored during development and should not be described as wholly untouched.
- These precision/recall/F1 values use greedy one-to-one matching per image and class at IoU 0.50 and do not replace COCO mAP.
- Transverse crack detections were not observed at the selected threshold.
- No large model checkpoints or dataset images are included in this publication.

The five-epoch experiment is a preliminary baseline, not a deployment-ready detector. Results are recorded as measured; no unverified improvement is claimed. The existing YOLO Streamlit application is not automatically replaced by this experiment.
