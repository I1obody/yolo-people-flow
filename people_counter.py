#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчет людей на видео с использованием YOLOv8/RT-DETR и трекингом.
Цель: посчитать увеличение потока людей (вход/выход через линию).
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
from datetime import datetime
import os

class PeopleCounter:
    def __init__(self, model_path='yolov8n.pt', conf=0.4, iou=0.5, line_y_ratio=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.line_y_ratio = line_y_ratio
        
        # Для трекинга
        self.tracks = {}
        self.next_id = 1
        self.counted_ids = set()
        
        # Статистика
        self.total_count = 0
        self.enter_count = 0
        self.exit_count = 0
        self.frame_counts = []
        
    def get_line_y(self, frame_height):
        return int(frame_height * self.line_y_ratio)
    
    def update_tracks(self, detections, frame_height):
        line_y = self.get_line_y(frame_height)
        
        # Простой трекинг по ближайшему соседу
        current_centers = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centers.append((cx, cy, det))
        
        # Сопоставление с существующими треками
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(current_centers)))
        
        for track_id, track_data in self.tracks.items():
            last_cx, last_cy = track_data['last_center']
            best_match = None
            best_dist = float('inf')
            
            for idx in unmatched_detections:
                cx, cy, _ = current_centers[idx]
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < best_dist and dist < 100:  # порог
                    best_dist = dist
                    best_match = idx
            
            if best_match is not None:
                cx, cy, det = current_centers[best_match]
                self.tracks[track_id]['last_center'] = (cx, cy)
                self.tracks[track_id]['last_y'] = cy
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['active'] = True
                
                # Проверка пересечения линии
                prev_y = self.tracks[track_id]['prev_y']
                curr_y = cy
                
                if prev_y < line_y <= curr_y and track_id not in self.counted_ids:
                    self.enter_count += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    print(f"Человек вошел: ID {track_id}")
                elif prev_y > line_y >= curr_y and track_id not in self.counted_ids:
                    self.exit_count += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    print(f"Человек вышел: ID {track_id}")
                
                self.tracks[track_id]['prev_y'] = curr_y
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(best_match)
        
        # Новые треки
        for idx in unmatched_detections:
            cx, cy, det = current_centers[idx]
            self.tracks[self.next_id] = {
                'last_center': (cx, cy),
                'last_y': cy,
                'prev_y': cy,
                'bbox': det['bbox'],
                'active': True
            }
            self.next_id += 1
        
        # Удаление неактивных треков
        tracks_to_remove = []
        for track_id in unmatched_tracks:
            if self.tracks[track_id]['active']:
                self.tracks[track_id]['active'] = False
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
    
    def process_frame(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': conf
                        })
        
        self.update_tracks(detections, frame.shape[0])
        self.frame_counts.append(len(detections))
        
        # Визуализация
        annotated = results[0].plot()
        
        # Рисуем линию подсчета
        h, w = frame.shape[:2]
        line_y = self.get_line_y(h)
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 0), 2)
        cv2.putText(annotated, f"Count Line", (10, line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Статистика на кадре
        stats_text = f"Total: {self.total_count} | Enter: {self.enter_count} | Exit: {self.exit_count} | Current: {len(detections)}"
        cv2.putText(annotated, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated, len(detections)
    
    def save_stats(self, output_path):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_count': self.total_count,
            'enter_count': self.enter_count,
            'exit_count': self.exit_count,
            'avg_people_per_frame': np.mean(self.frame_counts) if self.frame_counts else 0,
            'max_people_per_frame': max(self.frame_counts) if self.frame_counts else 0
        }
        
        df = pd.DataFrame([stats])
        df.to_csv(output_path, index=False)
        print(f"Статистика сохранена в {output_path}")
        return stats

def main():
    parser = argparse.ArgumentParser(description='Подсчет людей на видео')
    parser.add_argument('--video', type=str, required=True, help='Путь к видео')
    parser.add_argument('--output', type=str, default='output.mp4', help='Путь к выходному видео')
    parser.add_argument('--stats', type=str, default='stats.csv', help='Путь к файлу статистики')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Модель YOLO')
    parser.add_argument('--line_ratio', type=float, default=0.5, help='Положение линии подсчета (0-1)')
    parser.add_argument('--conf', type=float, default=0.4, help='Порог уверенности')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Ошибка: файл {args.video} не найден")
        return
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {args.video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    counter = PeopleCounter(model_path=args.model, conf=args.conf, line_y_ratio=args.line_ratio)
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, count = counter.process_frame(frame)
        out.write(annotated)
        
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Кадр {frame_num}: людей на кадре = {count}, всего подсчитано = {counter.total_count}")
        
        # Отображение
        cv2.imshow('People Counter', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    stats = counter.save_stats(args.stats)
    print("\n=== Итоговая статистика ===")
    print(f"Всего людей подсчитано: {stats['total_count']}")
    print(f"Вошло: {stats['enter_count']}")
    print(f"Вышло: {stats['exit_count']}")
    print(f"Среднее количество людей на кадре: {stats['avg_people_per_frame']:.2f}")

if __name__ == '__main__':
    main()
