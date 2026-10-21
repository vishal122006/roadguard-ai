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
# Pin the fallback to a known model revision so an upstream branch update
# cannot silently change which weights this app downloads.
BASELINE_REVISION = os.getenv(
    "ROADGUARD_BASELINE_REVISION",
    "7286e42df69b1b64a910f341f05b91e14b928e4b",
).strip()


def resolve_checkpoint() -> tuple[str, str, bool]:
    """Resolve a real checkpoint and identify whether it is the public fallback."""
    if CUSTOM_MODEL_ENV:
        custom_path = Path(CUSTOM_MODEL_ENV).expanduser()
        if custom_path.is_file():
            return str(custom_path), "Custom RoadGuard checkpoint", False
        print(
            "ROADGUARD_MODEL_PATH points to a missing checkpoint; "
            "trying the project checkpoint and public baseline.",
            flush=True,
        )

    if DEFAULT_MODEL_PATH.is_file():
        return str(DEFAULT_MODEL_PATH), "Project checkpoint (weights/best.pt)", False

    if not USE_PUBLIC_BASELINE:
        raise FileNotFoundError(
            "No usable local checkpoint was found and the public baseline is disabled. "
            "Add weights/best.pt, correct ROADGUARD_MODEL_PATH, or enable "
            "ROADGUARD_USE_PUBLIC_BASELINE=true."
        )

    # This is a real, publicly hosted road-damage model, not the user's
    # custom-trained checkpoint. It is used only if no local checkpoint exists.
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=BASELINE_REPO_ID,
        filename=BASELINE_FILENAME,
        revision=BASELINE_REVISION or None,
    )
    return (
        checkpoint_path,
        f"Public community baseline ({BASELINE_REPO_ID}, revision {BASELINE_REVISION[:7]})",
        True,
    )


@st.cache_resource
def load_model() -> tuple[YOLO, str, bool]:
    checkpoint_path, model_source, is_public_baseline = resolve_checkpoint()
    model = YOLO(checkpoint_path)
    return model, model_source, is_public_baseline


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
    st.warning(
        "ROADGUARD_MODEL_PATH points to a missing file. The app will continue by "
        "checking weights/best.pt and, if needed, load the public real-image model. "
        "The actual model source will be shown with the results."
    )
elif not DEFAULT_MODEL_PATH.is_file() and not CUSTOM_MODEL_ENV and USE_PUBLIC_BASELINE:
    st.warning(
        "Your own trained checkpoint was not found. When you click Detect, the app "
        "will download and use a real third-party road-damage model. This is not "
        "your custom-trained RoadGuard checkpoint, and its classes/performance differ."
    )
elif not DEFAULT_MODEL_PATH.is_file() and not CUSTOM_MODEL_ENV and not USE_PUBLIC_BASELINE:
    st.warning(
        "No checkpoint is available. Add weights/best.pt, set ROADGUARD_MODEL_PATH, "
        "or enable ROADGUARD_USE_PUBLIC_BASELINE=true."
    )

if st.button("Check model availability"):
    try:
        with st.spinner("Checking that the real model can be downloaded and loaded..."):
            health_model, health_source, health_is_fallback = load_model()
        st.success("Model loaded successfully. Real inference is ready.")
        st.write(f"**Model source:** {health_source}")
        names = getattr(health_model, "names", {})
        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names)]
        elif isinstance(names, (list, tuple)):
            class_names = list(names)
        else:
            class_names = []
        if class_names:
            st.write("**Classes:** " + ", ".join(str(name) for name in class_names))
        if health_is_fallback:
            st.info(
                "This is the public community baseline, not your own trained checkpoint. "
                "It has no dedicated manhole class."
            )
    except Exception as exc:
        print(
            "RoadGuard model availability check failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        st.error(
            f"Model check failed ({type(exc).__name__}). Check Manage app → Logs. "
            "No fake prediction was generated."
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
                with st.spinner(
                    "Loading the real model checkpoint and analyzing the image..."
                ):
                    model, model_source, is_public_baseline = load_model()
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
                if is_public_baseline:
                    st.warning(
                        "This third-party model has an 'Other' class that may include "
                        "manholes, patches, oil spills, and road markings. It does not "
                        "provide a dedicated manhole label."
                    )
                else:
                    st.caption(
                        "Class names are read from the loaded checkpoint. Make sure "
                        "they match the class labels used in your training dataset."
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
                    "Could not load or run the real detection model. "
                    f"Error type: {type(exc).__name__}. "
                    "Check Manage app → Logs for the detailed error. "
                    "No fallback or synthetic predictions were substituted."
                )
                if isinstance(exc, (ConnectionError, TimeoutError)):
                    st.info(
                        "The model download may have failed. Check that the app can "
                        "reach Hugging Face, then reboot the app and try again."
                    )
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {type(exc).__name__}.")

st.caption(
    "RoadGuard AI is an academic prototype and should not be used as the sole "
    "basis for road-safety or maintenance decisions. The public fallback is "
    "published by vinothvikas1987 on Hugging Face under Apache-2.0; review the "
    "upstream model card and dataset terms before redistribution or commercial use."
)
