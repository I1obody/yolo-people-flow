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
import subprocess
from tqdm import tqdm

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
        self.max_missed_frames = 10
        
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
        
        for track_id, track_data in list(self.tracks.items()):
            last_cx, last_cy = track_data['last_center']
            best_match = None
            best_dist = float('inf')
            
            for idx in unmatched_detections:
                cx, cy, _ = current_centers[idx]
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < best_dist and dist < 150:  # порог увеличен
                    best_dist = dist
                    best_match = idx
            
            if best_match is not None:
                cx, cy, det = current_centers[best_match]
                prev_y = self.tracks[track_id]['prev_y']
                curr_y = cy
                
                # Проверка пересечения линии ДО обновления prev_y
                # Вошедшие: с нижнего края (большие y) в верхний (меньшие y) - движение вверх
                # В координатах OpenCV y увеличивается вниз, поэтому движение вверх = уменьшение y
                if prev_y > line_y >= curr_y and track_id not in self.counted_ids:
                    self.enter_count += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    print(f"\nЧеловек вошел: ID {track_id}")
                elif prev_y < line_y <= curr_y and track_id not in self.counted_ids:
                    self.exit_count += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    print(f"\nЧеловек вышел: ID {track_id}")
                
                # Обновление трека
                self.tracks[track_id]['last_center'] = (cx, cy)
                self.tracks[track_id]['last_y'] = cy
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['prev_y'] = curr_y
                self.tracks[track_id]['missed_frames'] = 0
                self.tracks[track_id]['active'] = True
                
                unmatched_tracks.discard(track_id)
                unmatched_detections.discard(best_match)
            else:
                # Трекинг потерян, увеличиваем счетчик пропущенных кадров
                self.tracks[track_id]['missed_frames'] = self.tracks[track_id].get('missed_frames', 0) + 1
                if self.tracks[track_id]['missed_frames'] > self.max_missed_frames:
                    # Удаляем трек только после нескольких пропусков
                    if self.tracks[track_id]['active']:
                        self.tracks[track_id]['active'] = False
        
        # Новые треки
        for idx in unmatched_detections:
            cx, cy, det = current_centers[idx]
            self.tracks[self.next_id] = {
                'last_center': (cx, cy),
                'last_y': cy,
                'prev_y': cy,
                'bbox': det['bbox'],
                'active': True,
                'missed_frames': 0
            }
            self.next_id += 1
        
        # Удаление неактивных треков
        tracks_to_remove = []
        for track_id in list(self.tracks.keys()):
            if not self.tracks[track_id]['active'] and self.tracks[track_id].get('missed_frames', 0) > self.max_missed_frames:
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
        # Сохраняем в CSV с правильной кодировкой
        csv_path = output_path
        if not csv_path.endswith('.csv'):
            csv_path = csv_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Также сохраняем в Excel для корректного отображения
        excel_path = csv_path.replace('.csv', '.xlsx')
        df.to_excel(excel_path, index=False)
        
        print(f"Статистика сохранена в {csv_path} и {excel_path}")
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
    
    # Исправление FPS для AVI с некорректным avg_frame_rate
    # OpenCV может возвращать 1000 fps вместо реальных 30 для некоторых AVI
    def get_real_fps(cap, fps):
        # Если FPS слишком высокий (>120) - вероятно ошибка метаданных
        if fps > 120:
            try:
                # Пытаемся получить реальный FPS через ffprobe
                cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=r_frame_rate,avg_frame_rate,time_base,duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    args.video
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 4:
                        r_num, r_den = map(int, lines[0].split('/'))
                        a_num, a_den = map(int, lines[1].split('/'))
                        r_fps = r_num / r_den if r_den != 0 else fps
                        a_fps = a_num / a_den if a_den != 0 else fps
                        # Используем r_frame_rate если он разумен
                        if 1 <= r_fps <= 120:
                            print(f"Исправление FPS: OpenCV={fps}, r_frame_rate={r_fps}, avg_frame_rate={a_fps}")
                            return r_fps
                        # Если r_frame_rate не разумен, но есть duration и frame_count, вычислить FPS
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if frame_count > 0:
                            try:
                                duration = float(lines[3])
                                if duration > 0:
                                    calc_fps = frame_count / duration
                                    if 1 <= calc_fps <= 120:
                                        print(f"Исправление FPS по duration: OpenCV={fps}, calc_fps={calc_fps:.2f}, duration={duration}")
                                        return calc_fps
                            except:
                                pass
                        # Альтернативный расчет: используем time_base для вычисления FPS
                        try:
                            tb_num, tb_den = map(int, lines[2].split('/'))
                            time_base = tb_num / tb_den if tb_den != 0 else 1/1000
                            # Для AVI с time_base 1/1000 и frame_count, реальный FPS = frame_count / duration
                            # Но если duration недоступен, используем r_frame_rate
                            if r_fps > 120 or r_fps < 1:
                                # Попробуем вычислить из time_base и avg_frame_rate
                                if a_fps > 120:
                                    # Для AVI с avg_frame_rate=1000/1, реальный FPS обычно 30
                                    print(f"Обнаружен AVI с некорректным avg_frame_rate: {a_fps}, используем r_frame_rate={r_fps}")
                                    if 1 <= r_fps <= 120:
                                        return r_fps
                        except:
                            pass
            except Exception as e:
                print(f"ffprobe недоступен, используем эвристику: {e}")
        
        # Эвристика: если fps > 120, пробуем использовать 30 как дефолт
        if fps > 120:
            print(f"Предупреждение: FPS={fps} слишком высокий, используем 30 fps")
            return 30.0
        
        return fps
    
    fps = get_real_fps(cap, fps)
    
    # Определяем кодек в зависимости от формата выходного файла
    if args.output.lower().endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif args.output.lower().endswith('.mkv'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    counter = PeopleCounter(model_path=args.model, conf=args.conf, line_y_ratio=args.line_ratio)
    
    # Получаем реальную длительность видео
    try:
        cmd_duration = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            args.video
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=5)
        if result_duration.returncode == 0:
            video_duration = float(result_duration.stdout.strip())
            print(f"Длительность видео: {video_duration:.2f} сек")
        else:
            video_duration = None
    except:
        video_duration = None
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    opencv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Вычисляем реальное количество кадров
    if video_duration and video_duration > 0:
        real_frame_count = int(video_duration * fps)
        print(f"Реальное количество кадров: {real_frame_count}, OpenCV сообщает: {opencv_frame_count}")
    else:
        real_frame_count = opencv_frame_count
    
    # Фактор пропуска кадров из-за некорректного FPS в AVI
    skip_factor = 1
    if original_fps > 120 and video_duration and video_duration > 0:
        if opencv_frame_count > real_frame_count:
            skip_factor = int(opencv_frame_count / real_frame_count)
            print(f"Фактор пропуска кадров: {skip_factor}")
    
    # Skip frames для обработки (макс 30 FPS)
    # При skip_factor > 1, мы читаем каждый skip_factor-й кадр из OpenCV
    # чтобы получить реальный кадр
    skip_frames = max(1, int(fps / 30)) * skip_factor
    print(f"skip_frames: {skip_frames}")
    
    # Создаем прогресс-бар на основе реального количества кадров
    total_frames = real_frame_count
    print(f"Обработка видео: {total_frames} кадров")
    pbar = tqdm(total=total_frames, desc="Обработка", unit="кадр")
    
    frame_num = 0
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        if frame_num % skip_frames != 0:
            continue
        
        annotated, count = counter.process_frame(frame)
        # Записываем кадр skip_frames раз для сохранения длительности
        for _ in range(skip_frames):
            out.write(annotated)
        
        processed_frames += 1
        pbar.update(skip_frames)
        if processed_frames % 30 == 0:
            pbar.set_postfix({'людей': count, 'всего': counter.total_count})
        
        # Отображение (отключено для headless режима)
        # cv2.imshow('People Counter', annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    out.release()
    pbar.close()
    
    stats = counter.save_stats(args.stats)
    print("\n=== Итоговая статистика ===")
    print(f"Всего людей подсчитано: {stats['total_count']}")
    print(f"Вошло: {stats['enter_count']}")
    print(f"Вышло: {stats['exit_count']}")
    print(f"Среднее количество людей на кадре: {stats['avg_people_per_frame']:.2f}")

if __name__ == '__main__':
    main()
