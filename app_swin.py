"""Streamlit inference UI for a trained RoadGuard Swin-T detector.

Run only after training a real MMDetection checkpoint. Missing weights cause an
explicit error; no unrelated or synthetic model predictions are substituted.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="RoadGuard AI · Swin", page_icon="🛣️", layout="wide")
ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.getenv("ROADGUARD_CONFIG", ROOT / "configs/roadguard_swin_faster_rcnn.py")).expanduser()
CHECKPOINT = Path(os.getenv("ROADGUARD_CHECKPOINT", ROOT / "checkpoints/roadguard_swin.pth")).expanduser()
THRESHOLDS = Path(os.getenv("ROADGUARD_THRESHOLDS", ROOT / "checkpoints/best_thresholds.json")).expanduser()
CLASS_LABELS = {
    "D00": "Longitudinal crack", "D10": "Transverse crack",
    "D20": "Alligator crack", "D40": "Pothole",
}

st.title("RoadGuard AI")
st.subheader("Swin Transformer · multi-scale road-damage detection")
st.write("Upload a real road photograph and compare the original with detections from the trained model.")
st.info("Research prototype only. A missed detection does not prove a road is damage-free; do not use this output as the sole basis for safety decisions.")

@st.cache_resource(show_spinner="Loading Swin Transformer detector…")
def load_detector(config_path: str, checkpoint_path: str):
    import torch
    from mmdet.apis import init_detector
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return init_detector(config_path, checkpoint_path, device=device)

def read_thresholds():
    if not THRESHOLDS.is_file():
        return {}
    try:
        obj = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in obj.get("per_class_thresholds", {}).items()}
    except (OSError, ValueError, TypeError):
        return {}

def annotate(image, detections):
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, round(out.width / 48)))
    except OSError:
        font = ImageFont.load_default()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["label"]} {d["score"]:.2f}'
        width = max(2, round(min(out.size) / 240))
        draw.rectangle((x1, y1, x2, y2), outline=(30, 220, 95), width=width)
        b = draw.textbbox((x1, max(0, y1)), label, font=font)
        draw.rectangle((b[0], b[1], b[2] + 4, b[3] + 4), fill=(18, 110, 55))
        draw.text((b[0] + 2, b[1] + 1), label, fill="white", font=font)
    return out

uploaded = st.file_uploader("Upload a road photograph", type=["jpg", "jpeg", "png", "webp"])
conf = st.slider("Global confidence threshold", 0.05, 0.95, 0.25, 0.05)
use_calibrated = st.checkbox("Use per-class thresholds selected on validation data", value=True)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read image: {type(exc).__name__}: {exc}")
        st.stop()
    st.caption(f"Original dimensions: {image.width} × {image.height} pixels; inference uses the original image.")
    if min(image.size) < 480:
        st.warning("This image is small; fine cracks may not be resolvable.")
    if st.button("Detect road damage", type="primary"):
        if not CONFIG.is_file():
            st.error(f"MMDetection config not found: {CONFIG}")
            st.stop()
        if not CHECKPOINT.is_file():
            st.error(f"Trained Swin checkpoint not found: {CHECKPOINT}. Train the model first. No fallback predictions were generated.")
            st.stop()
        try:
            from mmdet.apis import inference_detector
            model = load_detector(str(CONFIG), str(CHECKPOINT))
            bgr = np.asarray(image)[:, :, ::-1].copy()  # MMDetection ndarray input expects BGR.
            with st.spinner("Running model inference…"):
                result = inference_detector(model, bgr)
            inst = result.pred_instances
            boxes = inst.bboxes.detach().cpu().numpy()
            scores = inst.scores.detach().cpu().numpy()
            labels = inst.labels.detach().cpu().numpy()
            cuts = read_thresholds() if use_calibrated else {}
            class_names = model.dataset_meta.get("classes", tuple(CLASS_LABELS))
            detections = []
            for box, score, class_id in zip(boxes, scores, labels):
                if int(class_id) < 0 or int(class_id) >= len(class_names):
                    continue
                code = str(class_names[int(class_id)])
                threshold = cuts.get(code, conf)
                if float(score) < threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                detections.append({
                    "class": code, "label": CLASS_LABELS.get(code, code),
                    "score": float(score),
                    "bbox": [max(0, x1), max(0, y1), min(image.width - 1, x2), min(image.height - 1, y2)],
                })
            annotated = annotate(image, detections)
            left, right = st.columns(2)
            with left:
                st.image(image, caption="Original image", use_container_width=True)
            with right:
                st.image(annotated, caption="Swin detections", use_container_width=True)
            st.write(f"Detections above the selected cutoff: **{len(detections)}**")
            if detections:
                st.dataframe([{
                    "Damage": d["label"], "Class": d["class"], "Score": round(d["score"], 4),
                    "Box [x1, y1, x2, y2]": [round(x, 1) for x in d["bbox"]],
                } for d in detections], use_container_width=True, hide_index=True)
            else:
                st.warning("No boxes passed the cutoff. This does not mean the road is damage-free.")
            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG")
            st.download_button("Download annotated result", buffer.getvalue(),
                               file_name="roadguard_swin_detections.png", mime="image/png")
            st.caption(f"Config: {CONFIG.name} · Checkpoint: {CHECKPOINT.name}")
        except Exception as exc:
            st.error(f"Inference failed ({type(exc).__name__}): {exc}")
            st.caption("Check the compatible PyTorch/CUDA/MMCV/MMEngine/MMDetection versions and checkpoint/config paths.")
