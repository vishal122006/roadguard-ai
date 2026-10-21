from pathlib import Path
import os

import streamlit as st

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
)

try:
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO
except Exception as exc:
    print(
        "RoadGuard AI inference dependency import failed: "
        f"{type(exc).__name__}: {exc}",
        flush=True,
    )
    st.title("🛣️ RoadGuard AI")
    st.error(
        "The YOLO inference runtime could not be loaded. "
        "Predictions are unavailable until the dependency issue is fixed. "
        "This app does not generate synthetic or demo predictions."
    )
    st.caption("Open Manage app → Logs to view the detailed import error.")
    st.stop()

APP_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = APP_DIR / "weights" / "best.pt"
CUSTOM_MODEL_ENV = os.getenv("ROADGUARD_MODEL_PATH", "").strip()
USE_PUBLIC_BASELINE = os.getenv("ROADGUARD_USE_PUBLIC_BASELINE", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
BASELINE_REPO_ID = os.getenv(
    "ROADGUARD_BASELINE_REPO_ID",
    "vinothvikas1987/pothole-detection-yolov8",
).strip()
BASELINE_FILENAME = os.getenv("ROADGUARD_BASELINE_FILENAME", "best.pt").strip()


def resolve_checkpoint() -> tuple[str, str]:
    """Find a real checkpoint; never substitute fabricated detections."""
    if CUSTOM_MODEL_ENV:
        custom_path = Path(CUSTOM_MODEL_ENV).expanduser()
        if not custom_path.is_file():
            raise FileNotFoundError(
                "ROADGUARD_MODEL_PATH is set, but the checkpoint does not exist: "
                f"{custom_path}"
            )
        return str(custom_path), "Custom RoadGuard checkpoint (ROADGUARD_MODEL_PATH)"

    if DEFAULT_MODEL_PATH.is_file():
        return str(DEFAULT_MODEL_PATH), "Project checkpoint (weights/best.pt)"

    if not USE_PUBLIC_BASELINE:
        raise FileNotFoundError(
            "No local checkpoint was found and the public baseline is disabled. "
            "Add weights/best.pt or set ROADGUARD_MODEL_PATH."
        )

    # Public community baseline trained on real road-damage data. This is a
    # fallback only; it is not represented as the project's own trained model.
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=BASELINE_REPO_ID,
        filename=BASELINE_FILENAME,
    )
    return (
        checkpoint_path,
        f"Public community baseline: {BASELINE_REPO_ID}/{BASELINE_FILENAME}",
    )


@st.cache_resource
def load_model() -> tuple[YOLO, str]:
    checkpoint_path, model_source = resolve_checkpoint()
    return YOLO(checkpoint_path), model_source


st.title("🛣️ RoadGuard AI")
st.subheader("Deep Learning-Based Road Damage Detection")
st.write(
    "Upload a real road photograph to detect visible road damage. "
    "The app displays model-produced bounding boxes, class names, and confidence scores."
)
st.info(
    "Research prototype only. Detections can be missed or incorrect; "
    "a 'no detections' result does not mean a road is damage-free."
)

if CUSTOM_MODEL_ENV and not Path(CUSTOM_MODEL_ENV).expanduser().is_file():
    st.error(
        "ROADGUARD_MODEL_PATH points to a missing file. Correct that setting or "
        "remove it to allow the public baseline fallback."
    )
elif not CUSTOM_MODEL_ENV and not DEFAULT_MODEL_PATH.is_file() and USE_PUBLIC_BASELINE:
    st.warning(
        "Your own trained checkpoint was not found. The app will use a real, "
        "publicly hosted road-damage baseline instead. Its labels and performance "
        "may differ from your custom RoadGuard model. It is not a synthetic/demo model."
    )
elif not CUSTOM_MODEL_ENV and not DEFAULT_MODEL_PATH.is_file() and not USE_PUBLIC_BASELINE:
    st.warning(
        "No checkpoint is available. Add weights/best.pt, set ROADGUARD_MODEL_PATH, "
        "or enable ROADGUARD_USE_PUBLIC_BASELINE=true."
    )

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded road photograph", width="stretch")

        if st.button("Detect road damage", type="primary"):
            try:
                with st.spinner("Loading the checkpoint and analyzing image..."):
                    model, model_source = load_model()
                    device = 0 if torch.cuda.is_available() else "cpu"
                    result = model.predict(
                        source=np.asarray(image),
                        imgsz=640,
                        conf=confidence,
                        device=device,
                        verbose=False,
                    )[0]

                annotated_rgb = result.plot()[:, :, ::-1]
                original_col, result_col = st.columns(2)
                with original_col:
                    st.image(image, caption="Original image", width="stretch")
                with result_col:
                    st.image(
                        annotated_rgb,
                        caption="Road-damage model predictions",
                        width="stretch",
                    )

                detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    detections.append(
                        {
                            "Damage type": result.names[class_id],
                            "Confidence": f"{score:.1%}",
                        }
                    )

                st.caption(f"Model source: {model_source}")
                st.caption(
                    "Class names come from the loaded checkpoint. The public baseline "
                    "uses road-damage categories including longitudinal/transverse/"
                    "alligator cracks, potholes, and an 'Other' category; 'Other' is "
                    "not a dedicated manhole label."
                )
                st.subheader("Detection summary")
                st.metric("Detected objects", len(detections))
                if detections:
                    st.dataframe(detections, width="stretch", hide_index=True)
                else:
                    st.warning(
                        "No objects were detected above this threshold. "
                        "This does not guarantee that the road is damage-free."
                    )
            except Exception as exc:
                print(
                    "RoadGuard AI inference failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                st.error(
                    "Model loading or inference failed. Check Manage app → Logs for "
                    "the detailed error. No fallback or synthetic predictions were substituted."
                )
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")

st.caption(
    "RoadGuard AI is an academic prototype and should not be used as the sole "
    "basis for road-safety or maintenance decisions. The public fallback model is "
    "provided by vinothvikas1987 on Hugging Face under its published Apache-2.0 license; "
    "review the model card and dataset terms before further redistribution or commercial use."
)
