# RoadGuard AI automatic Swin tuning

This branch now includes an automated validation-driven search for Swin-T + FPN + Faster R-CNN. It keeps the original YOLO app untouched.

## Files added
- `scripts/auto_tune_swin.py`: runs three candidate profiles, evaluates each on validation COCO mAP, chooses the best and fine-tunes it.
- `scripts/convert_voc_to_coco.py`: compatibility entry point for the XML-to-COCO converter.
- `scripts/check_dataset.py`: split/annotation/image integrity checks.
- `scripts/tune_thresholds.py`: entry point for validation-only class threshold tuning and frozen-threshold test F1.
- `scripts/benchmark_inference.py`: entry point for measuring inference latency, FPS, model size and parameter count.
- `notebooks/RoadGuard_SwinTransformer_AutoTune.ipynb`: guided Colab workflow.
- `configs/roadguard_swin_faster_rcnn.py`: Swin-T/FPN/Faster R-CNN model setup.

## Important
Run the notebook in a GPU-enabled Colab session. MMDetection's MMCV build must match the runtime PyTorch/CUDA version. If installation fails, restart the runtime and use the official MMDetection compatibility instructions.

The search ranks configurations on the validation split only. Confidence thresholds are then chosen on validation; the test split is evaluated only after choices are frozen. Precision may increase at the cost of recall, and no improvement is guaranteed until a full run completes. Do not report generated or unmeasured values as experimental results.
