# MII4_60_Python/core/visualizer.py

import numpy as np
import cv2

def load_colormap(filepath):
    """Загружает палитру из CSV файла."""
    hex_colors = np.loadtxt(filepath, dtype=str)
    # Конвертируем HEX в BGR (формат OpenCV)
    colormap = np.array([int(h[4:6], 16) for h in hex_colors]), \
               np.array([int(h[2:4], 16) for h in hex_colors]), \
               np.array([int(h[0:2], 16) for h in hex_colors])
    return np.stack(colormap, axis=1).astype(np.uint8)


def create_phase_image(phase_data, colormap, inverse=False):
    min_val, max_val = np.min(phase_data), np.max(phase_data)
    if max_val == min_val:
        return np.zeros((*phase_data.shape, 3), dtype=np.uint8)
    if inverse:
        min_val, max_val = max_val, min_val
    indices = (phase_data - min_val) / (max_val - min_val) * (len(colormap) - 1)
    indices = np.round(indices).astype(int)
    color_image = colormap[indices]
    return color_image

def create_phase_image_gray(phase_data, inverse=False):
    min_val, max_val = np.min(phase_data), np.max(phase_data)
    if max_val == min_val:
        return np.zeros(phase_data.shape, dtype=np.uint8)
    if inverse:
        min_val, max_val = max_val, min_val
    img = (phase_data - min_val) / (max_val - min_val)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def create_interferogram(images, method='average'):
    """
    Создает интерферограмму из набора изображений
    
    Args:
        images: список numpy arrays с изображениями
        method: метод создания ('average', 'first', 'last')
        
    Returns:
        numpy array с интерферограммой
    """
    if not images or len(images) == 0:
        return None
        
    if method == 'average':
        # Усредняем все изображения
        interferogram = np.mean(images, axis=0)
    elif method == 'first':
        # Используем первое изображение
        interferogram = images[0]
    elif method == 'last':
        # Используем последнее изображение
        interferogram = images[-1]
    else:
        # По умолчанию - усреднение
        interferogram = np.mean(images, axis=0)
    
    return interferogram.astype(np.uint8)

def save_data_to_csv(data, filepath):
    """
    Сохраняет 2D массив данных в CSV файл
    
    Args:
        data: 2D numpy array с данными
        filepath: путь к файлу для сохранения
    """
    import csv
    from datetime import datetime
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Записываем заголовок с метаданными
            writer.writerow([f"# Phase Data Export"])
            writer.writerow([f"# Timestamp: {datetime.now().isoformat()}"])
            writer.writerow([f"# Dimensions: {data.shape[1]}x{data.shape[0]}"])
            writer.writerow([])  # Пустая строка
            
            # Записываем данные
            for row in data:
                writer.writerow(row.tolist())
                
        return True
        
    except Exception as e:
        print(f"Ошибка сохранения CSV: {e}")
        return False