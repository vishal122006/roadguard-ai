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
    from PIL import Image, ImageDraw
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


# This second-stage model only analyses crops which the detector labeled "Other".
# These are candidate descriptions, not a new supervised road-damage detector.
ROAD_FEATURE_CANDIDATES = {
    "a pothole, a bowl-shaped hole in asphalt pavement": "Pothole",
    "a longitudinal crack running along an asphalt road": "Longitudinal crack",
    "a transverse crack running across an asphalt road": "Transverse crack",
    "interconnected alligator cracking on damaged asphalt": "Alligator crack",
    "a metal manhole cover embedded in a road": "Manhole cover",
    "a rectangular road repair patch in asphalt": "Road repair patch",
    "an oil or fluid stain on asphalt pavement": "Oil or fluid stain",
    "painted white or yellow road markings": "Road marking",
    "loose gravel, stones, or debris on a road": "Loose debris or gravel",
    "standing water or a puddle on a road surface": "Standing water or puddle",
    "broken or crumbling asphalt pavement": "Broken asphalt",
    "normal asphalt road surface without a visible defect": "Normal road surface",
}


@st.cache_resource
def load_visual_classifier():
    """Load a small zero-shot image classifier only when an Other box exists."""
    from transformers import pipeline

    return pipeline(
        task="zero-shot-image-classification",
        model="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        device=0 if torch.cuda.is_available() else -1,
    )


def classify_other_crop(crop: Image.Image):
    """Return best candidate and relative similarity for a detected Other crop."""
    classifier = load_visual_classifier()
    predictions = classifier(
        crop,
        candidate_labels=list(ROAD_FEATURE_CANDIDATES.keys()),
    )
    if not predictions:
        return None

    best = predictions[0]
    candidate = str(best["label"])
    return {
        "label": ROAD_FEATURE_CANDIDATES.get(candidate, candidate),
        "score": float(best["score"]),
        "alternatives": [
            {
                "label": ROAD_FEATURE_CANDIDATES.get(str(item["label"]), str(item["label"])),
                "score": float(item["score"]),
            }
            for item in predictions[:3]
        ],
    }


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

st.caption(
    "Checkpoint status: the app prefers your trained weights/ best.pt or "
    "ROADGUARD_MODEL_PATH. If neither is available, it loads a real public "
    "road-damage checkpoint when you request a model check or run detection. "
    "The model actually used is reported in the result."
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
        # Preserve the original PIL image for inference. The UI preview is shown
        # full-width; tiny source images trigger a warning because resizing cannot
        # recover detail that is not present in the uploaded file.
        st.caption(
            f"Source image: {image.width} × {image.height} pixels. "
            "Detection runs on the original uploaded pixels."
        )
        if image.width < 640 or image.height < 480:
            st.warning(
                "This upload is very small for road-damage analysis. "
                f"It is only {image.width} × {image.height} pixels, so fine cracks and "
                "small defects may be impossible to identify reliably. Upload the "
                "original camera photo or full-resolution image—not a thumbnail or screenshot."
            )
        st.image(
            image,
            caption="Uploaded road photograph",
            width="stretch",
        )
        st.download_button(
            "Download original image",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime=uploaded_file.type or "image/jpeg",
        )

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

                # Build per-detection labels. When the detector returns "Other",
                # use TinyCLIP on the cropped region to suggest a more specific type.
                # This is a visual estimate; its relative score is not calibrated
                # confidence and must not be presented as ground-truth classification.
                # Create an annotation canvas sized for screen viewing. This canvas
                # is only for display; all detection and crop classification use the
                # original image dimensions and unmodified pixels.
                image_width, image_height = image.size
                display_width = max(800, min(image_width, 1400))
                display_scale = display_width / max(1, image_width)
                display_height = max(1, int(round(image_height * display_scale)))
                annotated_image = image.resize(
                    (display_width, display_height),
                    resample=Image.Resampling.LANCZOS,
                )
                draw = ImageDraw.Draw(annotated_image)
                try:
                    from PIL import ImageFont
                    label_font = ImageFont.truetype(
                        "DejaVuSans.ttf",
                        size=max(14, min(24, int(18 * display_scale))),
                    )
                except Exception:
                    label_font = ImageFont.load_default()

                detections = []
                secondary_classifier_failed = False
                other_class_count = 0
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    detector_score = float(box.conf[0].item())
                    detector_class = str(result.names[class_id])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Keep crop in-bounds and add context around the detected region.
                    pad_x = max(6, int((x2 - x1) * 0.12))
                    pad_y = max(6, int((y2 - y1) * 0.12))
                    crop_box = (
                        max(0, x1 - pad_x),
                        max(0, y1 - pad_y),
                        min(image_width, x2 + pad_x),
                        min(image_height, y2 + pad_y),
                    )
                    crop = image.crop(crop_box)

                    is_other = detector_class.strip().lower() == "other"
                    likely_type = detector_class
                    visual_score = None
                    top_alternatives = []
                    if is_other:
                        other_class_count += 1
                        try:
                            classification = classify_other_crop(crop)
                            if classification:
                                likely_type = classification["label"]
                                visual_score = classification["score"]
                                top_alternatives = classification["alternatives"]
                            else:
                                likely_type = "Road feature (type unclear)"
                                secondary_classifier_failed = True
                        except Exception as exc:
                            print(
                                "Secondary road-feature classification failed: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            likely_type = "Road feature (type unclear)"
                            secondary_classifier_failed = True

                    if is_other and visual_score is not None:
                        if visual_score >= 0.22:
                            display_type = f"Likely {likely_type}"
                        else:
                            display_type = f"Possible {likely_type} (weak visual match)"
                    elif is_other:
                        display_type = likely_type
                    else:
                        display_type = detector_class

                    # Scale detector coordinates to the screen-friendly annotation canvas.
                    # Keep the label concise so it remains readable and doesn't run off-edge.
                    sx1 = int(round(x1 * display_scale))
                    sy1 = int(round(y1 * display_scale))
                    sx2 = int(round(x2 * display_scale))
                    sy2 = int(round(y2 * display_scale))
                    line_width = max(3, int(round(3 * display_scale)))
                    draw.rectangle(
                        (sx1, sy1, sx2, sy2),
                        outline=(255, 70, 70),
                        width=line_width,
                    )
                    short_type = likely_type
                    if is_other and visual_score is not None:
                        if visual_score < 0.22:
                            short_type = f"Possible {likely_type}"
                        else:
                            short_type = f"Likely {likely_type}"
                    annotation = f"{short_type} · {detector_score:.0%}"
                    if is_other and visual_score is not None:
                        annotation += f" · match {visual_score:.0%}"

                    pad = 5
                    label_bbox = draw.textbbox((0, 0), annotation, font=label_font)
                    text_w = label_bbox[2] - label_bbox[0]
                    text_h = label_bbox[3] - label_bbox[1]
                    # If the label is wider than the canvas, fall back to a short class.
                    if text_w > display_width - 2 * pad:
                        annotation = f"{short_type[:26]} · {detector_score:.0%}"
                        label_bbox = draw.textbbox((0, 0), annotation, font=label_font)
                        text_w = label_bbox[2] - label_bbox[0]
                        text_h = label_bbox[3] - label_bbox[1]

                    text_x = min(max(0, sx1), max(0, display_width - text_w - 2 * pad))
                    above_y = sy1 - text_h - 2 * pad
                    text_y = above_y if above_y >= 0 else min(
                        max(0, sy1 + 2), max(0, display_height - text_h - 2 * pad)
                    )
                    draw.rectangle(
                        (
                            text_x,
                            text_y,
                            min(display_width - 1, text_x + text_w + 2 * pad),
                            min(display_height - 1, text_y + text_h + 2 * pad),
                        ),
                        fill=(25, 25, 25),
                    )
                    draw.text(
                        (text_x + pad, text_y + pad),
                        annotation,
                        fill=(255, 255, 255),
                        font=label_font,
                    )

                    detections.append(
                        {
                            "Likely damage / feature": display_type,
                            "Original detector class": detector_class,
                            "Detector score": f"{detector_score:.1%}",
                            "Visual match score": (
                                f"{visual_score:.1%} (relative, not probability)"
                                if visual_score is not None
                                else "Not used"
                            ),
                            "_alternatives": top_alternatives,
                        }
                    )

                st.subheader("Original image")
                st.image(
                    image,
                    caption=f"Original upload ({image.width} × {image.height}px)",
                    width="stretch",
                )
                st.download_button(
                    "Download original",
                    data=uploaded_file.getvalue(),
                    file_name=uploaded_file.name,
                    mime=uploaded_file.type or "image/jpeg",
                    key="download_original_result",
                )
                st.subheader("Road damage detection output")
                st.image(
                    annotated_image,
                    caption="Full-width detection output. The display canvas is resized for readability; inference uses the original image.",
                    width="stretch",
                )

                st.caption(f"Model source: {model_source}")
                if other_class_count:
                    st.info(
                        "For detections originally labeled 'Other', a second image model "
                        "suggests a likely feature type from the cropped region. Its visual "
                        "match score is relative to the candidate list, not a calibrated "
                        "probability or a verified damage diagnosis."
                    )
                if secondary_classifier_failed:
                    st.warning(
                        "The secondary visual classifier could not load or run for at least "
                        "one region. Those entries are left as 'Road feature (type unclear)' "
                        "rather than guessing."
                    )

                st.subheader("Detection summary")
                st.metric("Detected objects", len(detections))
                if detections:
                    public_rows = [
                        {key: value for key, value in row.items() if key != "_alternatives"}
                        for row in detections
                    ]
                    st.dataframe(public_rows, width="stretch", hide_index=True)

                    for idx, row in enumerate(detections, start=1):
                        alternatives = row.get("_alternatives", [])
                        if alternatives:
                            with st.expander(
                                f"Detection {idx}: other possible visual matches"
                            ):
                                st.write(
                                    "The top candidates from the second-stage image model:"
                                )
                                st.dataframe(
                                    [
                                        {
                                            "Candidate": alt["label"],
                                            "Relative visual match": f'{alt["score"]:.1%}',
                                        }
                                        for alt in alternatives
                                    ],
                                    width="stretch",
                                    hide_index=True,
                                )
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
