#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация статистики потока людей
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def plot_stats(stats_file, output_dir='plots'):
    if not os.path.exists(stats_file):
        print(f"Файл {stats_file} не найден")
        return
    
    df = pd.read_csv(stats_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # График изменения количества людей
    plt.figure(figsize=(12, 6))
    
    # Если есть данные по кадрам, строим график
    if 'frame_counts' in df.columns:
        frame_data = df['frame_counts'].tolist()
        plt.plot(frame_data, label='Люди на кадре', alpha=0.7)
        plt.fill_between(range(len(frame_data)), frame_data, alpha=0.3)
    else:
        # Используем сохраненные данные
        print("Данные по кадрам не найдены, используем итоговую статистику")
    
    plt.title('Динамика количества людей на видео')
    plt.xlabel('Кадр')
    plt.ylabel('Количество людей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'people_dynamics.png'))
    plt.close()
    
    # Круговая диаграмма входа/выхода
    if 'enter_count' in df.columns and 'exit_count' in df.columns:
        plt.figure(figsize=(8, 8))
        labels = ['Вошло', 'Вышло']
        sizes = [df['enter_count'].iloc[0], df['exit_count'].iloc[0]]
        colors = ['#66b3ff', '#ff9999']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Распределение потока людей')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, 'flow_distribution.png'))
        plt.close()
    
    # Сводная статистика
    stats_text = f"""
Статистика подсчета людей
========================

Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Итоговые показатели:
- Всего подсчитано: {df['total_count'].iloc[0]}
- Вошло: {df['enter_count'].iloc[0]}
- Вышло: {df['exit_count'].iloc[0]}
- Среднее количество на кадре: {df['avg_people_per_frame'].iloc[0]:.2f}
- Максимум на кадре: {df['max_people_per_frame'].iloc[0]}

Коэффициент увеличения потока:
{df['enter_count'].iloc[0] / max(df['exit_count'].iloc[0], 1):.2f}
"""
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(stats_text)
    
    print(f"Визуализация сохранена в {output_dir}/")
    print(stats_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Визуализация статистики')
    parser.add_argument('--stats', type=str, default='stats.csv', help='Файл статистики')
    parser.add_argument('--output', type=str, default='plots', help='Папка для сохранения графиков')
    
    args = parser.parse_args()
    plot_stats(args.stats, args.output)
