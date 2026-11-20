"""Run the full pipeline on the labelled images and compare with ground truth.

Usage (from the repo root, e.g. in Colab after `pip install ultralytics`):
    python eval/run_eval.py --csv eval/ground_truth.csv
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from metrics import compute, fmt  # noqa: E402
from verdict import assign_ppe, decide  # noqa: E402


def load_truth(path):
    truth = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            truth[r["image"]] = {"people": int(r["people"]), "eq": int(r["fully_equipped"]),
                                 "ne": int(r["not_equipped"]), "un": int(r["unsure"])}
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "eval", "ground_truth.csv"))
    ap.add_argument("--images", default=os.path.join(ROOT, "eval", "images"))
    ap.add_argument("--settings", nargs="+", default=["640:0.40", "1280:0.20"],
                    help="imgsz:person_conf pairs")
    ap.add_argument("--ppe_conf", type=float, default=0.25)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "results.md"))
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    truth = load_truth(args.csv)
    ppe_model = YOLO(os.path.join(ROOT, "models", "best_n.pt"))
    person_model = YOLO("yolov8n.pt")

    lines = ["# Ground-truth evaluation", "",
             f"{len(truth)} hand-labelled images. People are compared by COUNT per image "
             "(not matched one to one), so these are approximate.", ""]
    for setting in args.settings:
        sz, pc = setting.split(":")
        sz, pc = int(sz), float(pc)
        pred = {}
        for name in truth:
            img = Image.open(os.path.join(args.images, name)).convert("RGB")
            ppe = ppe_model.predict(img, conf=args.ppe_conf, imgsz=sz, verbose=False)[0]
            per = person_model.predict(img, conf=pc, imgsz=sz, classes=[0], verbose=False)[0]
            items = [(ppe_model.names[int(c)], b.tolist())
                     for c, b in zip(ppe.boxes.cls, ppe.boxes.xyxy)]
            persons = [b.tolist() for b in per.boxes.xyxy]
            labels, _ = assign_ppe(items, persons)
            pred[name] = [decide(l)[0] for l in labels]
        m = compute(truth, pred)
        lines += [f"## imgsz {sz}, person conf {pc}", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {fmt(v)} |" for k, v in m.items()]
        lines.append("")
        print(f"--- imgsz {sz}, person conf {pc}")
        for k, v in m.items():
            print(f"{k:32} {fmt(v)}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written", args.out)


if __name__ == "__main__":
    main()
