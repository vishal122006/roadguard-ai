from pathlib import Path
import os

import streamlit as st

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
)

# Keep dependency failures inside the app UI so the page can explain the issue.
# The detailed import error remains available to the app owner in Cloud logs.
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
        "The real YOLO/OpenCV inference runtime could not be loaded. "
        "Predictions are unavailable until the dependency issue is fixed. "
        "This app does not generate synthetic or demo predictions."
    )
    st.caption("Open Manage app → Logs to view the detailed import error.")
    st.stop()

DEFAULT_MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"
MODEL_PATH = Path(os.getenv("ROADGUARD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {path}. Add weights/best.pt or set "
            "the ROADGUARD_MODEL_PATH environment variable."
        )
    return YOLO(str(path))


st.title("🛣️ RoadGuard AI")
st.subheader("Deep Learning-Based Road Damage Detection")
st.write(
    "Upload a road photograph to detect visible potholes, cracks, and manholes. "
    "The model returns predicted bounding boxes and confidence scores."
)
st.info(
    "Research prototype only. Predictions can be missed or incorrect; "
    "a 'no detections' result does not mean a road is damage-free."
)

if not MODEL_PATH.is_file():
    st.warning(
        "The trained checkpoint is not available. Real predictions are disabled "
        "until your actual best.pt file is made available at weights/best.pt "
        "or through ROADGUARD_MODEL_PATH. No synthetic/demo output will be used."
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

        if st.button(
            "Detect road damage",
            type="primary",
            disabled=not MODEL_PATH.is_file(),
        ):
            try:
                with st.spinner("Analyzing image..."):
                    model = load_model(str(MODEL_PATH))
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
                        caption="RoadGuard AI predictions",
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
                    "Inference failed. Check Manage app → Logs for the detailed "
                    "error. No fallback or synthetic predictions were substituted."
                )
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")

st.caption(
    "RoadGuard AI is an academic prototype and should not be used as the sole "
    "basis for road-safety or maintenance decisions."
)
