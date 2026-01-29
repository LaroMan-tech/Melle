import io
from pathlib import Path
from collections import Counter
import tempfile
import subprocess
import shutil

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import base64

from ultralytics import YOLO

# Standaard modelpad (gebruik deze automatisch als aanwezig)
DEFAULT_WEIGHTS_PATH = Path("models/best.pt")

# ----------------------------
# Helpers
# ----------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV (BGR)"""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCV (BGR) -> PIL (RGB)"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def box_iou(box_a, box_b) -> float:
    """IoU tussen twee xyxy-boxen."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom else 0.0


def center_distance(box_a, box_b) -> float:
    """Euclidische afstand tussen box-centers."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ca = ((xa1 + xa2) / 2.0, (ya1 + ya2) / 2.0)
    cb = ((xb1 + xb2) / 2.0, (yb1 + yb2) / 2.0)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def apply_class_conf_mask(result, per_class_conf, names):
    """Filter boxes per klasse threshold; fallback is geen filter."""
    if not per_class_conf or result.boxes is None or len(result.boxes) == 0:
        return result

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    mask = []
    for cid, cconf in zip(cls_ids, confs):
        cname = names.get(int(cid), str(cid))
        thr = per_class_conf.get(cname)
        if thr is None:
            thr = per_class_conf.get(str(cid))
        mask.append(cconf >= thr if thr is not None else True)

    mask = np.array(mask, dtype=bool)
    if mask.all():
        return result
    if not mask.any():
        result.boxes = result.boxes[:0]
        return result
    result.boxes = result.boxes[mask]
    return result


def draw_stairlift_path(frame_bgr: np.ndarray, dyn_pts, path_side="left", rail_main=(190, 190, 190), rail_edge=(110, 110, 110)):
    """Teken een traplift-rail en stoel; schaal afhankelijk van beeldgrootte."""
    if dyn_pts is None or len(dyn_pts) < 2:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    base = min(h, w) / 800.0  # schaalfactor rond 1 bij ~800px
    base = max(0.3, min(base, 3.0)) * 2  # verdubbel t.o.v. huidige schaal

    # verleng pad naar onderkant beeld (vloer) zodat rail start op vloer
    pts = np.array(dyn_pts, dtype=np.int32)
    start_x, start_y = pts[0]
    rail_edge_t = int(20 * base)
    rail_main_t = int(14 * base)
    if path_side == "right":
        start_x += rail_main_t
    elif path_side == "left":
        start_x -= rail_main_t
    pts = np.vstack([np.array([[start_x, h - int(10 * base)]]), pts])

    # Stoel op tweede of derde punt als beschikbaar
    seat_idx = 0
    if len(pts) >= 3:
        seat_idx = 2
    elif len(pts) >= 2:
        seat_idx = 1

    x, y = pts[seat_idx]
    seat_scale = 1.2  # 20% groter
    seat_w, seat_h = int(90 * base * seat_scale), int(55 * base * seat_scale)
    back_h = int(110 * base * seat_scale)
    arm_h = int(18 * base * seat_scale)
    leg_h = int(45 * base * seat_scale)

    # lichtbruin/beige stoel
    dark = (120, 170, 200)  # BGR warm lichtbruin
    edge = (90, 120, 140)
    accent = (255, 80, 70)
    glow = (80, 220, 255)

    overlay = frame_bgr.copy()
    # Rail (voorgrond t.o.v. pootjes)
    cv2.polylines(overlay, [pts], isClosed=False, color=rail_edge, thickness=rail_edge_t, lineType=cv2.LINE_AA)
    cv2.polylines(overlay, [pts], isClosed=False, color=rail_main, thickness=rail_main_t, lineType=cv2.LINE_AA)

    cv2.rectangle(overlay, (x - seat_w // 2, y - seat_h), (x + seat_w // 2, y), dark, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x - seat_w // 2, y - seat_h), (x + seat_w // 2, y), edge, max(1, int(2 * base)), cv2.LINE_AA)

    back_w = int(seat_w * 0.75)
    cv2.rectangle(overlay, (x - back_w // 2, y - seat_h - back_h), (x + back_w // 2, y - seat_h), dark, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x - back_w // 2, y - seat_h - back_h), (x + back_w // 2, y - seat_h), edge, max(1, int(2 * base)), cv2.LINE_AA)

    arm_w = int(seat_w * 0.48)
    cv2.rectangle(overlay, (x - arm_w, y - seat_h - arm_h), (x - arm_w + int(14 * base), y - seat_h + int(8 * base)), dark, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x + arm_w - int(14 * base), y - seat_h - arm_h), (x + arm_w, y - seat_h + int(8 * base)), dark, -1, cv2.LINE_AA)

    cv2.rectangle(overlay, (x - int(8 * base), y), (x + int(8 * base), y + leg_h), edge, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x - int(18 * base), y + leg_h), (x + int(18 * base), y + leg_h + int(10 * base)), dark, -1, cv2.LINE_AA)

    cv2.rectangle(overlay, (x + seat_w // 2 - int(6 * base), y - seat_h + int(5 * base)),
                  (x + seat_w // 2 - int(2 * base), y), glow, -1, cv2.LINE_AA)

    cv2.rectangle(overlay, (x + arm_w - int(12 * base), y - seat_h), (x + arm_w - int(4 * base), y - seat_h + int(12 * base)), accent, -1, cv2.LINE_AA)

    cv2.circle(overlay, (x, y), max(6, int(10 * base)), rail_edge, -1, cv2.LINE_AA)

   
    

    alpha = 0.9  # nog minder doorzichtig
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    # label "Stoel" bij de stoelbasis
    cv2.putText(frame_bgr, "Stoel", (x + int(seat_w * 0.55), y - int(seat_h * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * base, (200, 120, 0), max(1, int(2 * base)), cv2.LINE_AA)
    return frame_bgr


def compute_dynamic_path(cls_ids, boxes, names, target_names=None, path_side: str = "center"):
    """
    Bepaal een traplift-pad op basis van gedetecteerde treden/stootborden.
    Neemt centerpunten van geselecteerde klassen en sorteert op y (van onder naar boven).
    path_side: 'center' | 'left' | 'right' -> positie binnen de box
    """
    if target_names is None:
        target_names = {"trede", "tread", "step", "stair", "stootbord"}

    pts = []
    for cid, box in zip(cls_ids, boxes):
        cname = names.get(int(cid), str(cid)).lower()
        if cname in target_names:
            x1, y1, x2, y2 = box
            w = x2 - x1
            if path_side == "left":
                cx = x1 + 0.2 * w
            elif path_side == "right":
                cx = x2 - 0.2 * w
            else:
                cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            pts.append((cx, cy))

    if len(pts) < 2:
        return None

    # Sorteer van onder naar boven (grotere y eerst bij beeld-coords)
    pts_sorted = sorted(pts, key=lambda p: p[1], reverse=True)
    return np.array([(int(x), int(y)) for x, y in pts_sorted], dtype=np.int32)


def draw_guideline(frame_bgr: np.ndarray, side: str):
    """
    Teken een gebogen/diag lijn langs de trap. 'side' kan zijn:
    - 'none': niets
    - 'left_diag': van links-onder naar rechts-boven (zoals voorbeeld)
    - 'right_diag': van rechts-onder naar links-boven
    """
    if side == "none":
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    if side == "left_diag":
        norm_pts = [(0.05, 0.95), (0.20, 0.75), (0.50, 0.45), (0.75, 0.28), (0.90, 0.22)]
    elif side == "right_diag":
        norm_pts = [(0.95, 0.95), (0.80, 0.75), (0.50, 0.45), (0.25, 0.28), (0.10, 0.22)]
    else:
        return frame_bgr

    pts = np.array([(int(x * w), int(y * h)) for x, y in norm_pts], dtype=np.int32)
    cv2.polylines(frame_bgr, [pts], isClosed=False, color=(0, 0, 255), thickness=12, lineType=cv2.LINE_AA)
    return frame_bgr


def draw_guideline(frame_bgr: np.ndarray, side: str):
    """
    Teken een gebogen/diag lijn langs de trap. 'side' kan zijn:
    - 'none': niets
    - 'left_diag': van links-onder naar rechts-boven (zoals voorbeeld)
    - 'right_diag': van rechts-onder naar links-boven
    """
    if side == "none":
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    if side == "left_diag":
        norm_pts = [(0.05, 0.95), (0.20, 0.75), (0.50, 0.45), (0.75, 0.28), (0.90, 0.22)]
    elif side == "right_diag":
        norm_pts = [(0.95, 0.95), (0.80, 0.75), (0.50, 0.45), (0.25, 0.28), (0.10, 0.22)]
    else:
        return frame_bgr

    pts = np.array([(int(x * w), int(y * h)) for x, y in norm_pts], dtype=np.int32)
    cv2.polylines(frame_bgr, [pts], isClosed=False, color=(0, 0, 255), thickness=12, lineType=cv2.LINE_AA)
    return frame_bgr


@st.cache_resource
def load_model_from_bytes(file_bytes: bytes) -> YOLO:
    """
    Laad een YOLO .pt model vanuit bytes (Streamlit upload),
    door het tijdelijk weg te schrijven.
    """
    tmp_path = Path("temp_best.pt")
    tmp_path.write_bytes(file_bytes)
    return YOLO(str(tmp_path))


def run_inference(model: YOLO, pil_img: Image.Image, conf: float, iou: float, per_class_conf=None, path_side="left"):
    """
    Return:
      - plotted PIL image (with boxes)
      - counts dict {class_name: count}
      - raw detections list of dicts
    """
    bgr = pil_to_bgr(pil_img)

    results = model.predict(
        source=bgr,
        conf=conf,
        iou=iou,
        verbose=False
    )

    r = results[0]
    r = apply_class_conf_mask(r, per_class_conf or {}, model.names)
    names = model.names  # dict: id -> name

    counts = Counter()
    dets = []
    dyn_pts = None

    if r.boxes is not None and len(r.boxes) > 0:
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()

        for cid, cconf, box in zip(cls_ids, confs, xyxy):
            cname = names.get(int(cid), str(cid))
            counts[cname] += 1
            dets.append(
                {
                    "class_id": int(cid),
                    "class_name": cname,
                    "confidence": float(cconf),
                    "xyxy": [float(x) for x in box],
                }
            )

        dyn_pts = compute_dynamic_path(cls_ids, xyxy, names, path_side=path_side)

    # Gebruik originele frame; geen YOLO-boxlabels tekenen
    base_frame = r.orig_img.copy()
    overlayed = draw_stairlift_path(base_frame, dyn_pts, path_side=path_side)
    plotted_pil = bgr_to_pil(overlayed)

    return plotted_pil, dict(counts), dets


def run_inference_video(
    model: YOLO,
    video_bytes: bytes,
    filename: str,
    conf: float,
    iou: float,
    frame_skip: int = 3,
    unique_once: bool = False,
    iou_dedup: float = 0.7,
    center_thresh_ratio: float = 0.05,
    line_side: str = "none",
    dynamic_path: bool = False,
    path_side: str = "center",
    smooth_boxes: bool = True,
    smooth_alpha: float = 0.8,
    max_miss: int = 8,
    per_class_conf=None,
):
    """
    Verwerk een video: voorspel elk N-de frame en schrijf een geannoteerde mp4 terug.
    Return: (pad_annotated_video, counts_dict)
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Probeer H264 eerst (browser-vriendelijk), val terug op mp4v/MJPG.
    tmp_path = Path(tmp.name)
    writer = None
    out_path = None
    for cc, ext in [("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")]:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        candidate = tmp_path.with_name(tmp_path.stem + "_annotated" + ext)
        w = cv2.VideoWriter(str(candidate), fourcc, max(fps / frame_skip, 1), (width, height))
        if w.isOpened():
            writer = w
            out_path = candidate
            break
        w.release()

    if writer is None or out_path is None:
        progress.empty()
        st.error("Kon geen video-writer openen. Installeer ffmpeg/h264 codecs of probeer een andere omgeving.")
        cap.release()
        return None, {}

    total_counts = Counter()
    seen_boxes = []  # lijst van (class_id, box) die al geteld zijn (hele video)
    frame_id = 0
    progress = st.progress(0, text="Video wordt verwerkt...")
    prev_tracks = []  # lijst van dicts: {cid, box, conf, miss}
    prev_dyn_path = None
    floor_names = {"vloer", "floor"}
    counting_active = False
    floor_streak = 0
    no_floor_streak = 0

    for r in model.predict(source=tmp.name, conf=conf, iou=iou, stream=True, verbose=False):
        frame_id += 1
        if frame_id % frame_skip:
            continue

        # Klasse-specifieke confidence filter
        r = apply_class_conf_mask(r, per_class_conf or {}, model.names)

        names = model.names
        cls_ids = None
        xyxy = None
        if r.boxes:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            diag = (width**2 + height**2) ** 0.5
            max_center_dist = center_thresh_ratio * diag

            # trigger tellen op vloer
            floor_present = any(names.get(cid, str(cid)).lower() in floor_names for cid in cls_ids)
            if floor_present:
                floor_streak += 1
                no_floor_streak = 0
                if floor_streak == 1:
                    counting_active = True
            else:
                floor_streak = 0
                no_floor_streak += 1
                if no_floor_streak >= 1:
                    counting_active = False

            for cid, box in zip(cls_ids, xyxy):
                cname = names.get(cid, str(cid))
                if unique_once:
                    already_seen = False
                    for scid, sbox in seen_boxes:
                        if cid != scid:
                            continue
                        if box_iou(box, sbox) >= iou_dedup:
                            already_seen = True
                            break
                        if center_distance(box, sbox) <= max_center_dist:
                            already_seen = True
                            break
                    if already_seen:
                        continue
                    seen_boxes.append((cid, box))

                if counting_active:
                    if cname.lower() in floor_names:
                        if floor_streak <= 3:
                            total_counts[cname] += 1
                    else:
                        total_counts[cname] += 1

        # -----------------
        # Box smoothing
        # -----------------
        if smooth_boxes and cls_ids is not None and xyxy is not None:
            new_tracks = []
            used_prev = [False] * len(prev_tracks)

            for cid, box, cconf in zip(cls_ids, xyxy, r.boxes.conf.cpu().numpy()):
                # vind best match in prev_tracks met dezelfde klasse
                best_idx = -1
                best_iou = 0.0
                for i, t in enumerate(prev_tracks):
                    if used_prev[i]:
                        continue
                    if t["cid"] != cid:
                        continue
                    iou_val = box_iou(box, t["box"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = i

                if best_idx >= 0 and best_iou >= 0.5:
                    old = prev_tracks[best_idx]
                    used_prev[best_idx] = True
                    sm_box = smooth_alpha * old["box"] + (1 - smooth_alpha) * box
                    sm_conf = smooth_alpha * old["conf"] + (1 - smooth_alpha) * float(cconf)
                    new_tracks.append({"cid": cid, "box": sm_box, "conf": sm_conf, "miss": 0})
                else:
                    new_tracks.append({"cid": cid, "box": box, "conf": float(cconf), "miss": 0})

            # tracks die niet gematcht zijn: miss++
            for used, t in zip(used_prev, prev_tracks):
                if not used:
                    t["miss"] += 1
                    if t["miss"] <= max_miss:
                        new_tracks.append(t)

            prev_tracks = new_tracks
        else:
            # reset smoothing als we het niet doen
            prev_tracks = []

        # -----------------
        # Plot frame
        # -----------------
        frame_bgr = r.orig_img.copy()
        draw_tracks = prev_tracks if smooth_boxes else []
        # Labels/boxen niet tekenen voor clean view

        # Optionele hulplijn (diagonaal/gebogen) of dynamisch pad uit detecties
        if dynamic_path:
            dyn_pts = None
            if cls_ids is not None and xyxy is not None:
                dyn_pts = compute_dynamic_path(cls_ids, xyxy, names, path_side=path_side)
            if dyn_pts is None and prev_dyn_path is not None:
                dyn_pts = prev_dyn_path
            if dyn_pts is not None:
                cv2.polylines(frame_bgr, [dyn_pts], isClosed=False, color=(0, 0, 255), thickness=12, lineType=cv2.LINE_AA)
                prev_dyn_path = dyn_pts
        else:
            frame_bgr = draw_guideline(frame_bgr, line_side)

        writer.write(frame_bgr)
        if total_frames:
            progress.progress(min(frame_id / total_frames, 1.0))

    writer.release()
    cap.release()
    progress.empty()

    # Converteer altijd naar mp4/H264 (yuv420p) met ffmpeg als beschikbaar,
    # zodat de browser het zeker pakt, ongeacht wat de writer produceerde.
    final_path = out_path
    ffmpeg_path = shutil.which("ffmpeg")
    if out_path and ffmpeg_path:
        mp4_candidate = out_path.with_suffix(".mp4")
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(out_path),
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-acodec",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(mp4_candidate),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            final_path = mp4_candidate
        except subprocess.CalledProcessError:
            final_path = out_path

    return final_path, dict(total_counts)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Traplift app", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    :root {
        --brand-orange: #f49b00;
        --brand-blue: #36a9e0;
        --text-main: #1f2933;
        --panel: #f7f9fb;
        --border: #d9e2ec;
        --btn-bg: #111827;
        --btn-text: #ffffff;
    }
    body, .stApp { background: #fff; color: var(--text-main); }
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm { color: var(--text-main); }
    .stSlider label, .stSelectbox label, .stFileUploader label, .stCheckbox label, .stRadio label { color: #000000; font-weight: 600; }
    .stRadio div[role="radiogroup"] > label { color: #000000 !important; font-weight: 600; }
    .stRadio span { color: #000000 !important; }
    .stFileUploader, .stTextInput, .stSelectbox, .stNumberInput {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 10px;
    }
    .stFileUploader:hover, .stSelectbox:hover, .stTextInput:hover { border-color: var(--brand-blue); }
    .stSlider > div[data-baseweb="slider"] > div > div {
        background: var(--brand-orange);
    }
    .stSlider > div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(1) {
        background: linear-gradient(90deg, var(--brand-orange), var(--brand-blue));
    }
    .stSlider span { color: var(--text-main) !important; }
    .stButton button {
        background: var(--btn-bg);
        color: var(--btn-text);
        border: 1px solid #0b1220;
        border-radius: 6px;
    }
    .stButton button:hover { background: #1f2937; }
    .uploadedFile { color: var(--text-main); }

    /* File uploader buttons */
    .stFileUploader label div span { color: var(--btn-text) !important; }
    .stFileUploader label div { background: var(--btn-bg); color: var(--btn-text); }

    /* Expander label */
    .streamlit-expanderHeader p { color: var(--text-main) !important; }

    /* Video container fallback */
    .video-fallback { border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Traplift app")
st.write(
    "Ontdek uw traplift, gewoon bij u thuis\n\n"
    "Met deze app ziet u hoe een traplift er op uw eigen trap uit kan zien.\n\n "
    "Op basis van uw situatie ontvangt u een duidelijke en vrijblijvende prijsindicatie.\n\n"
    "Het uploaden van foto’s of een korte video van uw trap is eenvoudig en snel gedaan.\n\n "
    "Aan de hand daarvan kunnen wij een passend voorstel maken — u hoeft verder niets te doen.\n\n"
)

col_controls, col_main = st.columns([0.4, 0.6])

with col_main:
    promo_path = Path(r"C:\Users\melle\OneDrive - HvA\TBK Technische bedrijfskunde HvA\TBK jaar 3\Minor Data Science\Blok 2\Devi comfort case\UPstairs-42240-768x512.jpg")
    if promo_path.exists():
        st.markdown(
            f'<div style="text-align:right;"><img src="data:image/jpeg;base64,{base64.b64encode(promo_path.read_bytes()).decode()}" width="550"></div>',
            unsafe_allow_html=True,
        )

with col_controls:
    default_exists = DEFAULT_WEIGHTS_PATH.exists()
    use_default = default_exists
    weights_file = None

weights_bytes = None
weights_label = ""
if use_default and default_exists:
    weights_bytes = DEFAULT_WEIGHTS_PATH.read_bytes()
    weights_label = f"Standaard: {DEFAULT_WEIGHTS_PATH.name}"
elif weights_file is not None:
    weights_bytes = weights_file.read()
    weights_label = weights_file.name

if weights_bytes is None:
    st.info("Upload eerst een getraind YOLO-modelbestand (**best.pt**) of zorg dat het standaardmodel beschikbaar is.")
    st.stop()

# Laad model (cached)
model = load_model_from_bytes(weights_bytes)

# Vaste standaardinstellingen, geen UI-sliders
conf = 0.25
iou = 0.50
default_class_conf = {
    "Binnenbocht": 0.25,
    "Buitenbocht": 0.25,
    "Stootbord": 0.60,
    "Trede": 0.60,
    "Vloer": 0.15,
}
per_class_conf = {cname: default_class_conf.get(cname, conf) for cname in model.names.values()}

with col_controls:
    st.subheader("Uploads")
    st.markdown(
        """
**📸 Foto-instructie trap maken**

1. Start: ga onderaan de trap staan, ~1,5 m vóór de eerste trede.  
2. Camera: kies de breedste lens en richt verticaal recht op de trap.  
3. Eerste foto: overzicht van de hele trap vanaf de startpositie.  
4. Tweede foto: loop de trap op en maak een foto bij de eerste bocht (of bij de eerste paar treden bij een rechte trap).
        """
    )
    uploaded_files = st.file_uploader(
        "Foto's",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    st.markdown(
        """
**🎥 Video-instructie trap opnemen**

1. Start: ga onderaan de trap staan, ~1,5 m vóór de eerste trede.  
2. Camera: kies de breedste lens en richt verticaal recht op de trap.  
3. Filmen: start vóórdat je de trap oploopt.  
4. Oplopen: loop rustig omhoog, houd alle treden in beeld.  
5. Stop: beëindig wanneer je de eerste verdieping bereikt.
        """
    )
    video_file = st.file_uploader(
        "Video",
        type=["mp4", "mov", "avi", "webm"],
        key="video"
    )
    path_side_photo = st.radio(
        "Traplift spoor (foto)",
        ["Links", "Rechts"],
        index=0,
        horizontal=True,
        label_visibility="visible",
    )
    path_side_photo_val = "left" if path_side_photo == "Links" else "right"

if not uploaded_files and not video_file:
    st.info("Upload een of meerdere foto's of een video om te beginnen.")
    st.stop()

overall_counts = Counter()
photo_results = []

if uploaded_files:
    if len(uploaded_files) > 2:
        st.warning("Maximaal 2 foto's worden verwerkt; de rest wordt genegeerd.")
        uploaded_files = uploaded_files[:2]
    for uf in uploaded_files:
        img_bytes = uf.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)

        plotted_pil, counts, dets = run_inference(
            model,
            pil_img,
            conf=conf,
            iou=iou,
            per_class_conf=per_class_conf,
            path_side=path_side_photo_val,
        )
        overall_counts.update(counts)
        photo_results.append((uf.name, plotted_pil, counts, dets))

if photo_results:
    st.subheader("Resultaat foto's")
    for i in range(0, len(photo_results), 2):
        cols = st.columns(2)
        for col, item in zip(cols, photo_results[i:i+2]):
            name, img, counts, dets = item
            col.image(img, caption=name, use_container_width=True)

if video_file:
    st.divider()
    st.subheader(f"Video: {video_file.name}")
    video_bytes = video_file.read()

    # Gebruik vaste (stille) defaults voor een cleanere klantweergave
    unique_once = True
    line_side = "none"
    dynamic_path = True
    path_side = "left"  # volgt linkerkant standaard

    # Verlaag confidence specifiek voor trede/stootbord zodat ze vaker worden opgepikt in video
    per_class_conf_video = dict(per_class_conf)
    per_class_conf_video["Trede"] = 0.15
    per_class_conf_video["Stootbord"] = 0.20

    annotated_path, video_counts = run_inference_video(
        model,
        video_bytes,
        video_file.name,
        conf=conf,
        iou=iou,
        frame_skip=3,
        unique_once=unique_once,
        iou_dedup=0.7,
        center_thresh_ratio=0.05,
        line_side=line_side,
        dynamic_path=dynamic_path,
        path_side=path_side,
        per_class_conf=per_class_conf_video,
    )

    if annotated_path is None:
        st.error("De geannoteerde video kon niet worden opgeslagen.")
    else:
        st.success("Video klaar. Download de geannoteerde versie hieronder.")
        with open(annotated_path, "rb") as f:
            annotated_bytes = f.read()
            st.download_button(
                "Download annotated video",
                data=annotated_bytes,
                file_name=annotated_path.name,
                mime="video/mp4",
            )
        # Prijsindicatie op basis van video-detecties
        trede_v = video_counts.get("Trede", 0)
        stoot_v = video_counts.get("Stootbord", 0)
        base_cost = 3500 + 2000
        effective_units = 0.7 * stoot_v + 0.3 * trede_v
        variable_cost = effective_units * 100
        total_cost = base_cost + variable_cost

        st.markdown("**Prijsindicatie (op basis van video):**")
        st.write(f"Trede: {trede_v} · Stootbord: {stoot_v} · Effectief: {effective_units:.1f} x €100")
        st.write(f"Vaste kosten (stoel + elektra): €{base_cost:,.0f}".replace(',', '.'))
        st.write(f"Variabele kosten: €{variable_cost:,.0f}".replace(',', '.'))
        st.markdown(f"**Totaal indicatie: €{total_cost:,.0f}**".replace(',', '.'))
