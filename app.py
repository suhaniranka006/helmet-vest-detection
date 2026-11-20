"""Helmet & Safety Vest Detection - Streamlit demo (image upload, per-person verdict).

Two models are used:
  1. YOLOv8n (COCO, pretrained)  -> finds every *person*
  2. YOLOv8n fine-tuned by us    -> finds Helmet / Vest / No-Helmet / No-Vest
Each PPE box is assigned to the person whose box contains the PPE box's centre.
"""
import os

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

from verdict import center_inside, decide  # pure logic, unit tested in tests/

PPE_MODEL_PATH = "models/best_n.pt"   # our fine-tuned weights
PERSON_MODEL_PATH = "yolov8n.pt"      # COCO weights (auto-downloaded by ultralytics)
PERSON_CONF = 0.40
VERDICT_COLOR = {"SAFE": "green", "NOT SAFE": "red", "UNCERTAIN": "orange"}


@st.cache_resource            # models are loaded once, not on every click
def load_models():
    return YOLO(PPE_MODEL_PATH), YOLO(PERSON_MODEL_PATH)



st.set_page_config(page_title="Helmet & Vest Detection", layout="wide")
st.title("Helmet & Safety Vest Detection")
st.caption("YOLOv8n fine-tuned on a public Roboflow Universe dataset (CC BY 4.0). Demo project.")

conf = st.sidebar.slider("PPE confidence threshold", 0.05, 0.95, 0.25, 0.05,
                         help="Higher = fewer false alarms but more missed detections.")
st.sidebar.markdown(
    "**Rule (per person):** any *No-Helmet* / *No-Vest* = NOT SAFE. "
    "Helmet + Vest and no *No-* box = SAFE. Otherwise UNCERTAIN. "
    "PPE boxes outside every person box are ignored."
)

person_conf = st.sidebar.slider("Person confidence threshold", 0.05, 0.95, PERSON_CONF, 0.05)
imgsz = st.sidebar.selectbox("Inference image size", [640, 960, 1280], index=0,
                             help="Bigger = small / far-away people are caught more often, but slower.")

SAMPLE_DIR = "samples"   # free Pexels photos, so the demo works without uploading
samples = sorted(f for f in os.listdir(SAMPLE_DIR)) if os.path.isdir(SAMPLE_DIR) else []

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
sample = st.selectbox("...or pick a sample image", ["(none)"] + samples)

if file is not None:
    source = file
elif sample != "(none)":
    source = os.path.join(SAMPLE_DIR, sample)
else:
    st.info("Upload a photo of workers, or pick a sample, to see detections.")
    st.stop()

ppe_model, person_model = load_models()
image = Image.open(source).convert("RGB")

ppe_res = ppe_model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
person_res = person_model.predict(image, conf=person_conf, imgsz=imgsz, classes=[0], verbose=False)[0]

ppe_items = [(ppe_model.names[int(c)], b.tolist(), float(s))
             for c, b, s in zip(ppe_res.boxes.cls, ppe_res.boxes.xyxy, ppe_res.boxes.conf)]
persons = [b.tolist() for b in person_res.boxes.xyxy]

draw_img = image.copy()
draw = ImageDraw.Draw(draw_img)
rows, assigned = [], set()

for i, pbox in enumerate(persons, start=1):
    mine = [(n, b, s) for n, b, s in ppe_items if center_inside(b, pbox)]
    assigned.update(id(m[1]) for m in mine)
    verdict, why = decide({n for n, _, _ in mine})
    color = VERDICT_COLOR[verdict]
    draw.rectangle(pbox, outline=color, width=4)
    draw.text((pbox[0] + 4, max(pbox[1] - 14, 0)), f"Person {i}: {verdict}", fill=color)
    for n, b, s in mine:
        draw.rectangle(b, outline="white", width=2)
        draw.text((b[0] + 2, b[1] + 2), f"{n} {s:.2f}", fill="white")
    rows.append({"Person": i, "Verdict": verdict, "Detail": why})

left, right = st.columns(2)
left.image(image, caption="Original", use_container_width=True)
right.image(draw_img, caption="Detections (green=safe, red=not safe, orange=uncertain)",
            use_container_width=True)

ignored = sum(1 for _, b, _ in ppe_items if id(b) not in assigned)

if not persons:
    st.info("No person detected, so no verdict. "
            f"({len(ppe_items)} PPE-like box(es) were ignored.)")
else:
    bad = sum(r["Verdict"] == "NOT SAFE" for r in rows)
    ok = sum(r["Verdict"] == "SAFE" for r in rows)
    if bad:
        st.error(f"**NOT SAFE**: {bad} of {len(rows)} person(s) missing PPE")
    elif ok == len(rows):
        st.success(f"**SAFE**: all {len(rows)} person(s) wearing Helmet + Vest")
    else:
        st.warning("**UNCERTAIN**: some people could not be fully checked")
    st.table(rows)
    if ignored:
        st.caption(f"{ignored} PPE box(es) were outside every person box and ignored.")
