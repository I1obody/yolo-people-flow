#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетная обработка всех видео из ProdVideo с расчетом метрик
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
from people_counter import PeopleCounter
import cv2
from tqdm import tqdm

# Импорты для генерации отчетов
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False

def parse_video_filename(filename):
    """Парсит имя файла для извлечения даты"""
    # Формат: 1443 (2026-09-08 08'35'00 - 2026-09-08 09'05'00).avi
    pattern = r'\((\d{4}-\d{2}-\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def get_video_duration(video_path):
    """Получает длительность видео через ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None

def get_real_fps(cap, fps, video_path):
    """Определяет реальный FPS видео"""
    if fps > 120:
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,avg_frame_rate,time_base,duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 4:
                    r_num, r_den = map(int, lines[0].split('/'))
                    r_fps = r_num / r_den if r_den != 0 else fps
                    if 1 <= r_fps <= 120:
                        return r_fps
        except:
            pass
        return 30.0
    return fps

def process_video(video_path, output_dir):
    """Обрабатывает одно видео и возвращает статистику"""
    print(f"\n{'='*60}")
    print(f"Обработка: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = get_real_fps(cap, fps, video_path)
    duration = get_video_duration(video_path)
    
    print(f"FPS: {fps}, Разрешение: {width}x{height}, Длительность: {duration}s")
    
    # Создаем счетчик
    counter = PeopleCounter(model_path='yolov8n.pt', conf=0.4, line_y_ratio=0.5)
    
    # Обработка кадров
    frame_num = 0
    processed_frames = 0
    skip_frames = max(1, int(fps / 30))
    
    total_frames = int(duration * fps) if duration else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Обработка", unit="кадр")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        if frame_num % skip_frames != 0:
            continue
        
        # Обработка кадра
        results = counter.model(frame, conf=counter.conf, iou=counter.iou, verbose=False)
        
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
        
        counter.update_tracks(detections, frame.shape[0])
        counter.frame_counts.append(len(detections))
        
        processed_frames += 1
        pbar.update(skip_frames)
    
    cap.release()
    pbar.close()
    
    # Расчет метрик
    total_people = counter.enter_count + counter.exit_count
    entered = counter.enter_count
    exited = counter.exit_count
    
    # Всего людей в кадре (сумма по всем кадрам)
    total_people_in_frames = sum(counter.frame_counts)
    avg_people = np.mean(counter.frame_counts) if counter.frame_counts else 0
    max_people = max(counter.frame_counts) if counter.frame_counts else 0
    
    # Метрики
    # Коэффициент пропускной способности
    if total_people > 0:
        throughput_coeff = (entered / total_people) * 100
    else:
        throughput_coeff = 0
    
    # Коэффициент отказа (оттока)
    if total_people > 0:
        loss_coeff = (exited / total_people) * 100
    else:
        loss_coeff = 0
    
    # Индекс пиковой нагрузки
    if avg_people > 0:
        peak_index = max_people / avg_people
    else:
        peak_index = 0
    
    # Скорость обработки (Throughput per minute)
    throughput_per_min = entered / 30 if duration and duration > 0 else 0
    
    stats = {
        'video_file': os.path.basename(video_path),
        'date': parse_video_filename(os.path.basename(video_path)),
        'duration_sec': duration,
        'fps': fps,
        'total_frames': total_frames,
        'entered': entered,
        'exited': exited,
        'total_counted': total_people,
        'avg_people_per_frame': avg_people,
        'max_people_per_frame': max_people,
        'throughput_coeff': throughput_coeff,
        'loss_coeff': loss_coeff,
        'peak_index': peak_index,
        'throughput_per_min': throughput_per_min,
        'total_people_in_frames': total_people_in_frames
    }
    
    print(f"\nРезультаты для {os.path.basename(video_path)}:")
    print(f"  Вошедших: {entered}")
    print(f"  Вышедших: {exited}")
    print(f"  Коэффициент пропускной способности: {throughput_coeff:.2f}%")
    print(f"  Коэффициент отказа: {loss_coeff:.2f}%")
    print(f"  Индекс пиковой нагрузки: {peak_index:.2f}")
    print(f"  Скорость обработки: {throughput_per_min:.2f} чел/мин")
    
    return stats

def main():
    import numpy as np
    
    prod_video_dir = 'ProdVideo'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список видео
    video_files = []
    for file in os.listdir(prod_video_dir):
        if file.endswith('.avi') or file.endswith('.mp4'):
            video_files.append(os.path.join(prod_video_dir, file))
    
    video_files.sort()
    
    print(f"Найдено видео: {len(video_files)}")
    for f in video_files:
        print(f"  - {os.path.basename(f)}")
    
    all_stats = []
    
    for video_path in video_files:
        stats = process_video(video_path, output_dir)
        if stats:
            all_stats.append(stats)
    
    # Сохраняем результаты
    if all_stats:
        df = pd.DataFrame(all_stats)
        
        # Сохраняем в CSV
        csv_path = os.path.join(output_dir, 'video_stats.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Сохраняем в Excel
        excel_path = os.path.join(output_dir, 'video_stats.xlsx')
        df.to_excel(excel_path, index=False)
        
        # Сравнение дней
        print(f"\n{'='*80}")
        print("СРАВНЕНИЕ ДНЕЙ")
        print(f"{'='*80}")
        
        # Группируем по датам
        daily_stats = df.groupby('date').agg({
            'entered': 'sum',
            'exited': 'sum',
            'total_counted': 'sum',
            'throughput_coeff': 'mean',
            'loss_coeff': 'mean',
            'peak_index': 'mean',
            'throughput_per_min': 'mean',
            'avg_people_per_frame': 'mean',
            'max_people_per_frame': 'max'
        }).reset_index()
        
        print("\nСтатистика по дням:")
        print(daily_stats.to_string(index=False))
        
        # Сохраняем сводку по дням
        daily_csv = os.path.join(output_dir, 'daily_comparison.csv')
        daily_stats.to_csv(daily_csv, index=False, encoding='utf-8-sig')
        
        # Сравнение первых 3 дней с последними 3
        if len(daily_stats) >= 6:
            first_3 = daily_stats.head(3)
            last_3 = daily_stats.tail(3)
            
            print(f"\n{'='*80}")
            print("СРАВНЕНИЕ: Первые 3 дня vs Последние 3 дня")
            print(f"{'='*80}")
            
            comparison = {
                'metric': [],
                'first_3_avg': [],
                'last_3_avg': [],
                'change_pct': []
            }
            
            metrics = [
                ('entered', 'Вошедших'),
                ('exited', 'Вышедших'),
                ('throughput_coeff', 'Коэффициент пропускной способности (%)'),
                ('loss_coeff', 'Коэффициент отказа (%)'),
                ('peak_index', 'Индекс пиковой нагрузки'),
                ('throughput_per_min', 'Скорость обработки (чел/мин)'),
                ('avg_people_per_frame', 'Среднее кол-во людей в кадре')
            ]
            
            for metric_key, metric_name in metrics:
                first_avg = first_3[metric_key].mean()
                last_avg = last_3[metric_key].mean()
                change = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
                
                comparison['metric'].append(metric_name)
                comparison['first_3_avg'].append(first_avg)
                comparison['last_3_avg'].append(last_avg)
                comparison['change_pct'].append(change)
                
                print(f"\n{metric_name}:")
                print(f"  Первые 3 дня: {first_avg:.2f}")
                print(f"  Последние 3 дня: {last_avg:.2f}")
                print(f"  Изменение: {change:+.2f}%")
            
            comp_df = pd.DataFrame(comparison)
            comp_csv = os.path.join(output_dir, 'comparison_first_vs_last.csv')
            comp_df.to_csv(comp_csv, index=False, encoding='utf-8-sig')
            
            # Сохраняем сводку по дням в Excel
            daily_excel = os.path.join(output_dir, 'daily_comparison.xlsx')
            daily_stats.to_excel(daily_excel, index=False)
            
            # Сохраняем сравнение в Excel
            comp_excel = os.path.join(output_dir, 'comparison_first_vs_last.xlsx')
            comp_df.to_excel(comp_excel, index=False)
            
            print(f"\n{'='*80}")
            print(f"Результаты сохранены в папку {output_dir}/")
            print(f"  - video_stats.csv")
            print(f"  - video_stats.xlsx")
            print(f"  - daily_comparison.csv")
            print(f"  - daily_comparison.xlsx")
            print(f"  - comparison_first_vs_last.csv")
            print(f"  - comparison_first_vs_last.xlsx")
            print(f"{'='*80}")
            
            # Генерация дополнительных отчетов
            if REPORTS_AVAILABLE:
                print(f"\nГенерация дополнительных отчетов...")
                generate_reports(df, daily_stats, comp_df, output_dir)
            else:
                print(f"\nДля генерации PDF отчетов установите: pip install reportlab seaborn")

def generate_reports(video_stats_df, daily_stats_df, comparison_df, output_dir):
    """Генерация PDF отчета и тепловой карты"""
    try:
        # Регистрация шрифта
        font_path = 'C:/Windows/Fonts/arial.ttf'
        font_name = 'Arial'
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('Arial', font_path))
        else:
            font_name = 'Helvetica'
        
        # Создание графиков
        temp_plots_dir = os.path.join(output_dir, 'temp_plots')
        os.makedirs(temp_plots_dir, exist_ok=True)
        
        # График 1: Среднее количество людей в кадре
        plt.figure(figsize=(10, 6))
        plt.plot(daily_stats_df['date'], daily_stats_df['avg_people_per_frame'], marker='o', linewidth=2)
        plt.title('Среднее количество людей в кадре по дням', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Среднее кол-во людей в кадре')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plots_dir, 'avg_people.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 2: Вошедших/Вышедших
        plt.figure(figsize=(10, 6))
        plt.plot(daily_stats_df['date'], daily_stats_df['entered'], marker='o', label='Вошедших', linewidth=2)
        plt.plot(daily_stats_df['date'], daily_stats_df['exited'], marker='s', label='Вышедших', linewidth=2)
        plt.title('Вошедших и вышедших по дням', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Количество')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plots_dir, 'entered_exited.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # График 3: Коэффициенты
        plt.figure(figsize=(10, 6))
        plt.plot(daily_stats_df['date'], daily_stats_df['throughput_coeff'], marker='o', label='Пропускная способность (%)', linewidth=2)
        plt.plot(daily_stats_df['date'], daily_stats_df['loss_coeff'], marker='s', label='Коэффициент отказа (%)', linewidth=2)
        plt.title('Коэффициенты пропускной способности и отказа', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Процент')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plots_dir, 'coeffs.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Тепловая карта
        plt.figure(figsize=(12, 6))
        heatmap_data = daily_stats_df.set_index('date')[['avg_people_per_frame']]
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                    cbar_kws={'label': 'Среднее кол-во людей в кадре'},
                    linewidths=0.5)
        plt.title('Тепловая карта среднего количества людей в кадре по дням')
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'heatmap_avg_people_per_frame.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохраняем также в temp для PDF
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                    cbar_kws={'label': 'Среднее кол-во людей в кадре'},
                    linewidths=0.5)
        plt.title('Тепловая карта среднего количества людей в кадре по дням')
        plt.tight_layout()
        plt.savefig(os.path.join(temp_plots_dir, 'heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Создание PDF
        pdf_path = os.path.join(output_dir, 'video_analysis_report.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        for style_name in ['Normal', 'Heading1', 'Heading2', 'Heading3']:
            if style_name in styles:
                styles[style_name].fontName = font_name
        
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=1, fontSize=18, spaceAfter=30, fontName=font_name))
        styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], fontSize=14, spaceBefore=20, spaceAfter=12, fontName=font_name))
        
        story = []
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph('Отчет по анализу видеонаблюдения', styles['CenterTitle']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f'Дата генерации: {datetime.now().strftime("%d.%m.%Y %H:%M")}', styles['Normal']))
        story.append(Paragraph(f'Период анализа: {daily_stats_df["date"].min()} - {daily_stats_df["date"].max()}', styles['Normal']))
        story.append(PageBreak())
        
        # Таблица video_stats
        story.append(Paragraph('Сводная статистика по видео', styles['SectionTitle']))
        video_stats_display = video_stats_df[['video_file', 'date', 'entered', 'exited', 'total_counted', 'avg_people_per_frame']].copy()
        video_stats_display.columns = ['Видео файл', 'Дата', 'Вошедших', 'Вышедших', 'Всего', 'Среднее в кадре']
        table_data = [list(video_stats_display.columns)] + video_stats_display.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph('Динамика среднего количества людей в кадре', styles['SectionTitle']))
        story.append(Image(os.path.join(temp_plots_dir, 'avg_people.png'), width=6*inch, height=3.5*inch))
        story.append(PageBreak())
        
        # Ежедневное сравнение
        story.append(Paragraph('Ежедневное сравнение метрик', styles['SectionTitle']))
        daily_display = daily_stats_df.copy()
        daily_display.columns = ['Дата', 'Вошедших', 'Вышедших', 'Всего', 'Пропускная %', 'Отказ %', 'Пик индекс', 'Скорость чел/мин', 'Среднее в кадре', 'Макс в кадре']
        table_data = [list(daily_display.columns)] + daily_display.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph('Вошедших и вышедших', styles['SectionTitle']))
        story.append(Image(os.path.join(temp_plots_dir, 'entered_exited.png'), width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph('Коэффициенты', styles['SectionTitle']))
        story.append(Image(os.path.join(temp_plots_dir, 'coeffs.png'), width=6*inch, height=3.5*inch))
        story.append(PageBreak())
        
        # Сравнение периодов
        story.append(Paragraph('Сравнение первого и последнего периода', styles['SectionTitle']))
        comparison_display = comparison_df.copy()
        comparison_display.columns = ['Метрика', 'Среднее первые 3', 'Среднее последние 3', 'Изменение %']
        table_data = [list(comparison_display.columns)] + comparison_display.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph('Тепловая карта среднего количества людей в кадре', styles['SectionTitle']))
        story.append(Image(os.path.join(temp_plots_dir, 'heatmap.png'), width=6*inch, height=3*inch))
        
        story.append(PageBreak())
        story.append(Paragraph('Выводы', styles['SectionTitle']))
        story.append(Paragraph(f'За период с {daily_stats_df["date"].min()} по {daily_stats_df["date"].max()} проанализировано {len(video_stats_df)} видеофайлов.', styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f'Среднее количество вошедших: {daily_stats_df["entered"].mean():.1f} человек/день', styles['Normal']))
        story.append(Paragraph(f'Среднее количество вышедших: {daily_stats_df["exited"].mean():.1f} человек/день', styles['Normal']))
        story.append(Paragraph(f'Средняя пропускная способность: {daily_stats_df["throughput_coeff"].mean():.1f}%', styles['Normal']))
        story.append(Paragraph(f'Средний коэффициент отказа: {daily_stats_df["loss_coeff"].mean():.1f}%', styles['Normal']))
        
        doc.build(story)
        
        # Очистка временных файлов
        import shutil
        if os.path.exists(temp_plots_dir):
            shutil.rmtree(temp_plots_dir)
        
        print(f"  - video_analysis_report.pdf")
        print(f"  - heatmap_avg_people_per_frame.png")
        
    except Exception as e:
        print(f"Ошибка при генерации отчетов: {e}")

if __name__ == '__main__':
    main()
