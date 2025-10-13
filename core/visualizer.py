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
    """Преобразует 2D массив фазы в цветное изображение."""
    min_val, max_val = np.min(phase_data), np.max(phase_data)
    if max_val == min_val:
        return np.zeros((*phase_data.shape, 3), dtype=np.uint8)

    if inverse:
        min_val, max_val = max_val, min_val
        
    # Нормализуем данные в диапазон индексов палитры
    indices = (phase_data - min_val) / (max_val - min_val) * (len(colormap) - 1)
    indices = np.round(indices).astype(int)
    
    # Создаем цветное изображение используя палитру как lookup table
    color_image = colormap[indices]
    
    return color_image