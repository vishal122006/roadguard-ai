from pathlib import Path
import csv
import json
import os

import streamlit as st

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--rg-bg:#0a0e0d;--rg-panel:#111815;--rg-panel2:#151e18;--rg-line:#29372e;--rg-text:#f1f5ef;--rg-muted:#9ba99f;--rg-lime:#c7f36b;--rg-teal:#66d9e8;--rg-orange:#ffb86b;--rg-red:#ff7188}
html,body,[class*="css"]{font-family:'Manrope',system-ui,sans-serif}
.stApp{background:radial-gradient(ellipse at 84% 0%,rgba(199,243,107,.07),transparent 28%),var(--rg-bg);color:var(--rg-text)}
[data-testid="stAppViewContainer"]{background:transparent}
[data-testid="stHeader"]{background:rgba(10,14,13,.7)}
.block-container{max-width:1500px;padding-top:1.7rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{background:#0d1310;border-right:1px solid var(--rg-line)}
section[data-testid="stSidebar"] *{color:var(--rg-text)}
h1,h2,h3,h4{color:var(--rg-text)!important;letter-spacing:-.045em!important}
p,label,[data-testid="stMarkdownContainer"]{color:var(--rg-text)}
[data-testid="stMarkdownContainer"] a{color:var(--rg-lime)}
[data-testid="stFileUploaderDropzone"]{background:#101714;border:1px dashed #536b54;border-radius:16px}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--rg-lime);background:#151e17}
.stButton>button,.stDownloadButton>button{background:var(--rg-lime);color:#10170d;border:0;font-weight:800;border-radius:10px;min-height:2.75rem;transition:transform .15s ease,filter .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{color:#10170d;filter:brightness(1.06);transform:translateY(-1px);border:0}
.stSlider [role="slider"]{background:var(--rg-lime);border-color:var(--rg-lime)}
[data-testid="stMetric"]{background:linear-gradient(145deg,#151e18,#101512);border:1px solid var(--rg-line);border-radius:15px;padding:15px 17px}
[data-testid="stMetricLabel"]{color:var(--rg-muted)!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.09em}
[data-testid="stMetricValue"]{color:var(--rg-text)!important;font-weight:800;letter-spacing:-.05em}
[data-testid="stDataFrame"]{border:1px solid var(--rg-line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid var(--rg-line);background:#101613;border-radius:12px}
hr{border-color:var(--rg-line)!important}
.rg-hero{position:relative;overflow:hidden;border:1px solid #304033;border-radius:24px;padding:30px 32px 27px;background:linear-gradient(112deg,rgba(199,243,107,.085),transparent 47%),linear-gradient(145deg,#151e18,#101513 75%);margin-bottom:22px}
.rg-hero:after{content:'RG / 04';position:absolute;right:25px;bottom:4px;font:500 clamp(3.5rem,8vw,8rem) 'DM Mono',monospace;letter-spacing:-.12em;color:rgba(199,243,107,.05);pointer-events:none}
.rg-eyebrow{font:500 .68rem 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.17em;color:var(--rg-lime)}
.rg-hero h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:.98;margin:.7rem 0 .8rem;max-width:900px;font-weight:800}
.rg-hero p{color:#a8b5ab;max-width:790px;line-height:1.65;margin:0;font-size:.96rem}
.rg-chip{display:inline-flex;align-items:center;gap:8px;border:1px solid #465c3c;background:#182219;color:var(--rg-lime);border-radius:999px;padding:7px 11px;font:500 .66rem 'DM Mono',monospace;letter-spacing:.08em;text-transform:uppercase}
.rg-dot{width:7px;height:7px;border-radius:50%;background:var(--rg-lime);display:inline-block;box-shadow:0 0 12px #c7f36b80}
.rg-section{font:500 .68rem 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.15em;color:var(--rg-lime);margin:.5rem 0}
.rg-note{border:1px solid #54442d;border-radius:12px;padding:12px 15px;background:#1c1913;color:#f3d8a9;line-height:1.55;font-size:.82rem}
.rg-footer{border-top:1px solid var(--rg-line);margin-top:30px;padding-top:14px;color:#78857b;font:500 .64rem 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.1em}
@media(max-width:800px){.rg-hero{padding:23px 20px}.rg-hero:after{display:none}}
</style>
""", unsafe_allow_html=True)

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


st.markdown("""
<div class="rg-hero">
  <div class="rg-chip"><span class="rg-dot"></span> Road intelligence console · Computer vision</div>
  <h1>See the damage.<br>Measure the road.</h1>
  <p>RoadGuard AI turns real road photographs into inspectable damage detections, confidence scores and exportable evidence. Built for research workflows—not black-box claims.</p>
</div>
""", unsafe_allow_html=True)

header_left, header_right = st.columns([1.5, 1], gap="large")
with header_left:
    st.markdown('<div class="rg-section">01 / Inspection workspace</div>', unsafe_allow_html=True)
    st.markdown("### Road damage, made visible")
    st.write("Upload a full-resolution road photograph. Inspect model-produced boxes side-by-side with the original, review confidence scores and export the detections.")
with header_right:
    st.markdown('<div class="rg-section">System notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="rg-note"><b>Research prototype</b><br>False positives and missed damage are possible. A no-detection result does not prove that a road is damage-free.</div>', unsafe_allow_html=True)
    st.caption("The active checkpoint is identified in each result. Public baseline and project-trained weights are not interchangeable.")


def _load_evaluation_metrics():
    """Load real dataset-level metrics from an evaluation JSON or Ultralytics results.csv."""
    explicit = os.getenv("ROADGUARD_EVAL_REPORT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        APP_DIR / "evaluation_metrics.json",
        APP_DIR / "metrics.json",
        APP_DIR / "results" / "evaluation_metrics.json",
        APP_DIR / "results" / "metrics.json",
        APP_DIR / "outputs" / "test_metrics.json",
    ]
    for path in candidates:
        if not path or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for nested_key in ("metrics", "results", "test_metrics", "box"):
                    if isinstance(data.get(nested_key), dict):
                        data = {**data, **data[nested_key]}
                return data, str(path)
        except (OSError, json.JSONDecodeError):
            continue

    csv_paths = []
    for pattern in ("runs/detect/**/results.csv", "runs/**/results.csv", "results.csv"):
        csv_paths.extend(APP_DIR.glob(pattern))
    csv_paths = [p for p in set(csv_paths) if p.is_file()]
    for path in sorted(csv_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                return rows[-1], str(path)
        except (OSError, csv.Error):
            continue
    return {}, None


def _metric_value(data, candidates):
    normalized = {str(k).strip().lower().replace(" ", "").replace("_", "").replace("-", ""): v for k, v in data.items()}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "").replace("-", "")
        if key not in normalized:
            continue
        try:
            value = float(normalized[key])
            if value == value and abs(value) != float("inf"):
                return value * 100 if abs(value) <= 1 else value
        except (TypeError, ValueError):
            pass
    return None


def _show_metric(value):
    return f"{value:.2f}%" if value is not None else "Not recorded"


st.markdown('<div class="rg-section">02 / Current model accuracy & performance</div>', unsafe_allow_html=True)

SWIN_TRACKING_PATH = APP_DIR / "results" / "swin_tracking.json"

@st.fragment(run_every="60s")
def _render_swin_tracking():
    """Refresh the research metric card from the small JSON report every minute."""
    try:
        tracking = json.loads(SWIN_TRACKING_PATH.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("Tracking report must be a JSON object.")
    except FileNotFoundError:
        st.warning("No Swin tracking report is available yet. The dashboard will display metrics after a tracking report is committed.")
        return
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.warning(f"Could not read Swin tracking report: {exc}")
        return

    value = tracking.get("metric_value_percent")
    target = tracking.get("target_percent", 70.0)
    model_name = str(tracking.get("model", "Swin-Tiny + FPN + Faster R-CNN"))
    epoch = tracking.get("epoch")
    stage = str(tracking.get("stage", "baseline"))
    metric_name = str(tracking.get("metric_name", "COCO mAP@0.50:0.95 (validation)"))
    fine_tuned_evaluated = bool(tracking.get("fine_tuned_evaluated", False))
    split = str(tracking.get("split", "validation")).title()
    last_updated = str(tracking.get("last_updated_utc", "")).strip()

    try:
        value = float(value) if value is not None else None
        target = float(target)
        if value is not None and not (0 <= value <= 100):
            raise ValueError("metric_value_percent must be between 0 and 100.")
        if not (0 < target <= 100):
            raise ValueError("target_percent must be between 0 and 100.")
    except (TypeError, ValueError) as exc:
        st.warning(f"Swin tracking report contains invalid metric values: {exc}")
        return

    metric_cols = st.columns([1.15, 2.5, 1.1], gap="large")
    with metric_cols[0]:
        st.metric(metric_name, f"{value:.2f}%" if value is not None else "Not recorded",
                  help="Measured dataset-level detection metric from the saved Swin training report. It is mAP, not generic classification accuracy.")
    with metric_cols[1]:
        epoch_text = f", epoch {epoch}" if epoch is not None else ""
        st.markdown(f"**Current {stage.replace('_', ' ')}: {model_name}{epoch_text}**")
        if not fine_tuned_evaluated:
            st.write("The fine-tuned model has not yet been evaluated; no improvement is being claimed.")
        else:
            st.write(f"Latest saved result is from the {split.lower()} evaluation. Review held-out test results before claiming generalization.")
        if last_updated:
            st.caption(f"Report timestamp (UTC): {last_updated}")
    with metric_cols[2]:
        st.metric("Target mAP goal", f"{target:.0f}%",
                  help="Project target, not an achieved or guaranteed result.")

    if value is not None:
        progress = max(0.0, min(1.0, value / target))
        st.progress(progress, text=f"{progress * 100:.1f}% of the target value — progress indicator only, not a second accuracy score.")
        if value >= target:
            st.success("The validation metric meets the target. Confirm performance on the held-out test set.")
        else:
            st.caption(f"Remaining to target: {max(0.0, target - value):.2f} percentage points.")
    st.caption("This Swin research tracker is separate from the checkpoint used for photo detections. It rereads the GitHub-deployed JSON report every 60 seconds; updating the report in GitHub triggers the Streamlit deployment to refresh its data.")

_render_swin_tracking()

st.markdown('<div class="rg-section">03 / Detailed evaluation metrics</div>', unsafe_allow_html=True)
_eval_data, _eval_source = _load_evaluation_metrics()
_p = _metric_value(_eval_data, ["metrics/precision(B)", "precision", "box_precision", "P"])
_r = _metric_value(_eval_data, ["metrics/recall(B)", "recall", "box_recall", "R"])
_map50 = _metric_value(_eval_data, ["metrics/mAP50(B)", "mAP50", "map50", "AP50"])
_map5095 = _metric_value(_eval_data, ["metrics/mAP50-95(B)", "mAP50-95", "map50-95", "AP"])
_f1 = (2 * _p * _r / (_p + _r)) if _p is not None and _r is not None and (_p + _r) else None
score_cols = st.columns(5, gap="small")
for col, label, value, help_text in [
    (score_cols[0], "Precision", _p, "Of predicted detections, the share that are correct under the evaluation protocol."),
    (score_cols[1], "Recall", _r, "Of annotated objects, the share successfully detected."),
    (score_cols[2], "F1 score", _f1, "Harmonic mean of the reported precision and recall."),
    (score_cols[3], "mAP @ 0.50", _map50, "Mean average precision at IoU 0.50."),
    (score_cols[4], "mAP @ 0.50:0.95", _map5095, "COCO-style mean AP over IoU thresholds 0.50–0.95."),
]:
    with col:
        st.metric(label, _show_metric(value), help=help_text)
if _eval_source:
    st.caption(f"Dataset-level metrics loaded from: {_eval_source}. Values depend on the checkpoint and evaluation split used to create that report.")
else:
    st.caption("Dataset-level scores are not attached to this deployment yet. Add a real evaluation report as evaluation_metrics.json or set ROADGUARD_EVAL_REPORT; no placeholder scores are shown.")

st.markdown('<div class="rg-section">04 / Detection controls</div>', unsafe_allow_html=True)

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
                    # Use a larger detector input to retain more small-crack detail.
                    # The original uploaded pixels are passed through; no aspect-ratio
                    # distortion or synthetic super-resolution is applied.
                    result = model.predict(
                        source=np.asarray(image),
                        imgsz=960,
                        conf=confidence,
                        device=device,
                        verbose=False,
                    )[0]

                # Build per-detection labels. When the detector returns "Other",
                # use TinyCLIP on the cropped region to suggest a more specific type.
                # This is a visual estimate; its relative score is not calibrated
                # confidence and must not be presented as ground-truth classification.
                # Put both comparison images on the same standard 1280×720 canvas.
                # Keep the source aspect ratio (letterbox instead of stretching) and use
                # high-quality LANCZOS resampling for display. Resizing does not create
                # details absent from a low-resolution upload, so we still warn on tiny files.
                image_width, image_height = image.size
                canvas_width, canvas_height = 1280, 720
                # Keep the annotation-coordinate bounds aligned to this output canvas.
                # These aliases are used below when clipping label text to image edges.
                display_width, display_height = canvas_width, canvas_height
                display_scale = min(
                    canvas_width / max(1, image_width),
                    canvas_height / max(1, image_height),
                )
                resized_width = max(1, int(round(image_width * display_scale)))
                resized_height = max(1, int(round(image_height * display_scale)))
                offset_x = (canvas_width - resized_width) // 2
                offset_y = (canvas_height - resized_height) // 2
                display_image = image.resize(
                    (resized_width, resized_height),
                    resample=Image.Resampling.LANCZOS,
                )
                input_canvas = Image.new(
                    "RGB", (canvas_width, canvas_height), color=(18, 18, 18)
                )
                input_canvas.paste(display_image, (offset_x, offset_y))
                annotated_image = input_canvas.copy()
                draw = ImageDraw.Draw(annotated_image)
                # Labels are drawn on a 1280px canvas that may be scaled down
                # by Streamlit (especially on mobile). Use a bold, fixed-size font
                # so names like "Alligator Crack" remain legible after display scaling.
                try:
                    from PIL import ImageFont
                    font_candidates = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                        "DejaVuSans-Bold.ttf",
                    ]
                    label_font = None
                    for font_path in font_candidates:
                        try:
                            label_font = ImageFont.truetype(font_path, size=34)
                            break
                        except OSError:
                            continue
                    if label_font is None:
                        try:
                            label_font = ImageFont.load_default(size=30)
                        except TypeError:
                            label_font = ImageFont.load_default()
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
                    sx1 = int(round(x1 * display_scale)) + offset_x
                    sy1 = int(round(y1 * display_scale)) + offset_y
                    sx2 = int(round(x2 * display_scale)) + offset_x
                    sy2 = int(round(y2 * display_scale)) + offset_y
                    # Keep boxes thick enough to see after the canvas is displayed
                    # at a smaller width in the browser.
                    line_width = 5
                    draw.rectangle(
                        (sx1, sy1, sx2, sy2),
                        outline=(255, 60, 60),
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

                st.subheader("Input vs. detection output")
                input_col, output_col = st.columns(2, gap="medium")
                with input_col:
                    st.markdown("**Original input**")
                    st.image(
                        input_canvas,
                        caption=(
                            f"Source: {image.width} × {image.height}px · "
                            "fit to 1280 × 720 without stretching"
                        ),
                        width="stretch",
                    )
                    st.download_button(
                        "Download original",
                        data=uploaded_file.getvalue(),
                        file_name=uploaded_file.name,
                        mime=uploaded_file.type or "image/jpeg",
                        key="download_original_result",
                    )
                with output_col:
                    st.markdown("**Model output**")
                    st.image(
                        annotated_image,
                        caption=(
                            "Same 1280 × 720 comparison canvas with predicted boxes. "
                            "Original aspect ratio is preserved."
                        ),
                        width="stretch",
                    )
                st.caption(
                    "Both views use the same standard canvas for direct comparison. "
                    "Letterboxing preserves shape and avoids stretching. Upscaling can "
                    "make a preview larger, but cannot restore missing source detail."
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

st.markdown("""
<div class="rg-footer">ROADGUARD AI · COMPUTER VISION RESEARCH PROTOTYPE · VERIFY BEFORE OPERATIONAL USE</div>
""", unsafe_allow_html=True)
st.caption(
    "RoadGuard AI is an academic prototype and should not be used as the sole "
    "basis for road-safety or maintenance decisions. The public fallback is "
    "published by vinothvikas1987 on Hugging Face under Apache-2.0; review the "
    "upstream model card and dataset terms before redistribution or commercial use."
)
