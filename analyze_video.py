#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze video to understand people movement
"""
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('testvideo/884447251053.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = int(height * 0.5)

print(f'Video: {width}x{height}, FPS={fps}, line_y={line_y}')

frame_num = 0
people_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.4, iou=0.5, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({'cx': cx, 'cy': cy})
    
    if len(detections) > 0 and frame_num in range(140, 200):
        print(f'\nFrame {frame_num}: {len(detections)} people')
        for i, d in enumerate(detections):
            print(f'  Person {i}: cx={d["cx"]}, cy={d["cy"]}, relative to line: {d["cy"] - line_y:+d}')
    
    frame_num += 1
    if frame_num > 250:
        break

cap.release()
