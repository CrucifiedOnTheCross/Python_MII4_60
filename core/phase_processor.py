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

    def scale_phase(self, phase_radians):
        return phase_radians * (self.lambda_nm / (2 * np.pi))

    def threshold_unwrap(self, height_map, threshold=0.8, iterations=1, horizontal=True, vertical=True):
        if height_map is None:
            return None
        h = height_map.copy()
        lam = self.lambda_nm
        for _ in range(max(1, iterations)):
            if horizontal:
                rows, cols = h.shape
                for y in range(rows):
                    lastJ = h[y, 0]
                    dh = 0.0
                    for x in range(1, cols):
                        J = h[y, x]
                        if J - lastJ > 0.5 * lam * threshold:
                            dh -= 0.5 * lam
                        elif J - lastJ < -0.5 * lam * threshold:
                            dh += 0.5 * lam
                        h[y, x] = J + dh
                        lastJ = J
            if vertical:
                rows, cols = h.shape
                for x in range(cols):
                    lastJ = h[0, x]
                    dh = 0.0
                    for y in range(1, rows):
                        J = h[y, x]
                        if J - lastJ > 0.5 * lam * threshold:
                            dh -= 0.5 * lam
                        elif J - lastJ < -0.5 * lam * threshold:
                            dh += 0.5 * lam
                        h[y, x] = J + dh
                        lastJ = J
        return h

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

    def remove_polynomial_trend(self, phase_data):
        """
        Удаляет полиномиальный тренд из фазовых данных (аналог sphereTrend из Java версии)
        
        Args:
            phase_data: 2D numpy array с фазовыми данными
            
        Returns:
            2D numpy array с удаленным полиномиальным трендом
        """
        if phase_data is None or phase_data.size == 0:
            return phase_data
            
        height, width = phase_data.shape
        result = phase_data.copy()
        
        # Извлекаем горизонтальный и вертикальный профили
        horizontal_profile = result[0, :]  # Первая строка
        vertical_profile = result[:, 0]    # Первый столбец
        
        # Вычисляем полиномиальные тренды
        horizontal_trend = self._fit_polynomial_trend(horizontal_profile, width)
        vertical_trend = self._fit_polynomial_trend(vertical_profile, height)
        
        # Удаляем горизонтальный тренд
        for y in range(height):
            result[y, :] -= horizontal_trend
            
        # Удаляем вертикальный тренд
        for x in range(width):
            result[:, x] -= vertical_trend
            
        return result
    
    def _fit_polynomial_trend(self, profile, size):
        """
        Вычисляет полиномиальный тренд второго порядка для одномерного профиля
        
        Args:
            profile: 1D numpy array
            size: размер профиля
            
        Returns:
            1D numpy array с полиномиальным трендом
        """
        # Создаем матрицу A для полинома второго порядка: y = w1 + w2*x + w3*x^2
        A = np.zeros((size, 3))
        for i in range(size):
            A[i, 0] = 1      # константа
            A[i, 1] = i      # линейный член
            A[i, 2] = i * i  # квадратичный член
        
        # Решаем систему методом наименьших квадратов: A^T * A * w = A^T * y
        AT = A.T
        ATA = np.dot(AT, A)
        
        try:
            # Вычисляем обратную матрицу
            ATA_inv = np.linalg.inv(ATA)
            ATy = np.dot(AT, profile)
            coefficients = np.dot(ATA_inv, ATy)
            
            # Вычисляем полиномиальный тренд
            trend = np.zeros(size)
            for i in range(size):
                trend[i] = coefficients[0] + coefficients[1] * i + coefficients[2] * i * i
                
            return trend
            
        except np.linalg.LinAlgError:
            # Если матрица вырожденная, возвращаем нулевой тренд
            return np.zeros(size)