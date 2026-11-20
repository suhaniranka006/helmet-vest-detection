"""Helmet & Safety Vest Detection - Gradio demo for Hugging Face Spaces.

Same logic as the Streamlit app (app.py in the repo root); only the screen code differs.
Expects verdict.py, models/best_n.pt and samples/ next to this file.
"""
import os

import gradio as gr
from PIL import Image, ImageDraw
from ultralytics import YOLO

from verdict import assign_ppe, decide

HERE = os.path.dirname(os.path.abspath(__file__))
PPE_MODEL = YOLO(os.path.join(HERE, "models", "best_n.pt"))   # fine-tuned by me
PERSON_MODEL = YOLO("yolov8n.pt")                              # COCO, auto-downloaded
COLOR = {"SAFE": "green", "NOT SAFE": "red", "UNCERTAIN": "orange"}
SAMPLE_DIR = os.path.join(HERE, "samples")


def analyse(image, ppe_conf, person_conf, imgsz):
    if image is None:
        return None, "Upload a photo or click an example.", []
    image = image.convert("RGB")
    imgsz = int(imgsz)

    ppe = PPE_MODEL.predict(image, conf=ppe_conf, imgsz=imgsz, verbose=False)[0]
    per = PERSON_MODEL.predict(image, conf=person_conf, imgsz=imgsz, classes=[0], verbose=False)[0]
    items = [(PPE_MODEL.names[int(c)], b.tolist(), float(s))
             for c, b, s in zip(ppe.boxes.cls, ppe.boxes.xyxy, ppe.boxes.conf)]
    persons = [b.tolist() for b in per.boxes.xyxy]

    labels_per_person, ignored = assign_ppe([(n, b) for n, b, _ in items], persons)

    out = image.copy()
    draw = ImageDraw.Draw(out)
    rows = []
    for i, (pbox, labels) in enumerate(zip(persons, labels_per_person), start=1):
        verdict, why = decide(labels)
        draw.rectangle(pbox, outline=COLOR[verdict], width=4)
        draw.text((pbox[0] + 4, max(pbox[1] - 14, 0)), f"Person {i}: {verdict}", fill=COLOR[verdict])
        rows.append([i, verdict, why])
    for name, box, score in items:
        draw.rectangle(box, outline="white", width=2)
        draw.text((box[0] + 2, box[1] + 2), f"{name} {score:.2f}", fill="white")

    if not persons:
        msg = f"**No person detected, so no verdict.** ({len(items)} PPE-like box(es) were ignored.)"
    else:
        bad = sum(r[1] == "NOT SAFE" for r in rows)
        ok = sum(r[1] == "SAFE" for r in rows)
        if bad:
            msg = f"## NOT SAFE\n{bad} of {len(rows)} person(s) missing PPE."
        elif ok == len(rows):
            msg = f"## SAFE\nAll {len(rows)} person(s) wearing Helmet + Vest."
        else:
            msg = "## UNCERTAIN\nSome people could not be fully checked."
        if ignored:
            msg += f"\n\n{ignored} PPE box(es) were outside every person box and ignored."
    return out, msg, rows


examples = []
if os.path.isdir(SAMPLE_DIR):
    examples = [[os.path.join(SAMPLE_DIR, f)] for f in sorted(os.listdir(SAMPLE_DIR))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

with gr.Blocks(title="Helmet & Safety Vest Detection") as demo:
    gr.Markdown(
        "# Helmet & Safety Vest Detection\n"
        "YOLOv8n fine-tuned on a public Roboflow Universe dataset (CC BY 4.0). "
        "A second pretrained model finds each person; every helmet / vest box is assigned to a person. "
        "**Rule:** any *No-Helmet* / *No-Vest* = NOT SAFE; Helmet + Vest = SAFE; otherwise UNCERTAIN. "
        "This is a learning project, not a safety system. "
        "[Code, results and limitations on GitHub](https://github.com/suhaniranka006/helmet-vest-detection)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Photo")
            ppe_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="PPE confidence threshold")
            person_conf = gr.Slider(0.05, 0.95, value=0.40, step=0.05, label="Person confidence threshold")
            imgsz = gr.Radio([640, 960, 1280], value=640, label="Inference image size (bigger finds small / far people, slower)")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detections (green safe, red not safe, orange uncertain)")
            out_msg = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["Person", "Verdict", "Detail"], interactive=False)
    btn.click(analyse, [inp, ppe_conf, person_conf, imgsz], [out_img, out_msg, out_tbl])
    if examples:
        gr.Examples(examples=examples, inputs=[inp], label="Sample photos (Pexels)")

if __name__ == "__main__":
    demo.launch()
