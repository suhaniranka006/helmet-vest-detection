---
title: Helmet and Safety Vest Detection
emoji: 🦺
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
---

# Helmet & Safety Vest Detection

Upload a photo (or click an example) and the app finds each person and says whether that person
is wearing a helmet and a high-visibility vest: **SAFE / NOT SAFE / UNCERTAIN**.

A pretrained YOLOv8n finds people. A YOLOv8n fine-tuned by me on a public Roboflow Universe
dataset (OAKDev "vest", CC BY 4.0) finds Helmet, No-Helmet, Vest and No-Vest. Each PPE box is
assigned to the person whose box contains its centre.

This is a learning project, not a safety system. Results, evaluation, limitations and code:
https://github.com/suhaniranka006/helmet-vest-detection

Sample photos: Pexels (free to use under the Pexels License).
