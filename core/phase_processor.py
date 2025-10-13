# MII4_60_Python/core/phase_processor.py

import numpy as np
from skimage.restoration import unwrap_phase

class PhaseProcessor:
    """Выполняет все вычисления, связанные с фазой."""
    def __init__(self, lambda_nm=632.8):
        self.lambda_nm = lambda_nm

    def compute_phase(self, images, steps):
        """Вычисляет 'свёрнутую' фазу на основе серии изображений."""
        # Убедимся, что изображения в float для вычислений
        images = [img.astype(np.float32) for img in images]
        
        I = images
        numerator = np.zeros_like(I[0])
        denominator = np.zeros_like(I[0])

        if steps == 3:
            numerator = np.sqrt(3) * (I[2] - I[1])
            denominator = 2 * I[0] - I[1] - I[2]
        elif steps == 4:
            numerator = I[0] - I[2]
            denominator = I[3] - I[1]
        elif steps == 5:
            numerator = 2 * (I[1] - I[3])
            denominator = 2 * I[2] - I[0] - I[4]
        else:
            raise ValueError("Поддерживаются только 3, 4, или 5 шагов.")
            
        # Избегаем деления на ноль
        wrapped_phase = np.arctan2(numerator, denominator)
        return wrapped_phase

    def unwrap_phase(self, wrapped_phase):
        """Выполняет развёртку фазы."""
        # Используем готовую, быструю функцию из scikit-image
        return unwrap_phase(wrapped_phase)

    def remove_linear_trend(self, phase_data):
        """Удаляет линейный тренд (наклон) с изображения."""
        rows, cols = phase_data.shape
        
        # Удаляем тренд по строкам
        x = np.arange(cols)
        data_no_trend_rows = np.zeros_like(phase_data)
        for r in range(rows):
            p = np.polyfit(x, phase_data[r, :], 1)
            trend = np.polyval(p, x)
            data_no_trend_rows[r, :] = phase_data[r, :] - trend
            
        # Удаляем тренд по столбцам
        y = np.arange(rows)
        final_data = np.zeros_like(phase_data)
        for c in range(cols):
            p = np.polyfit(y, data_no_trend_rows[:, c], 1)
            trend = np.polyval(p, y)
            final_data[:, c] = data_no_trend_rows[:, c] - trend
            
        return final_data