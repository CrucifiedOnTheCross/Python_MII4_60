# MII4_60_Python/core/phase_processor.py

import numpy as np
from skimage.restoration import unwrap_phase

class PhaseProcessor:
    """Выполняет все вычисления, связанные с фазой."""
    def __init__(self, lambda_angstrom=7500.0):
        self.lambda_angstrom = lambda_angstrom
        self.lambda_nm = lambda_angstrom / 10.0

    def compute_phase(self, images, steps):
        """
        Вычисляет 'свёрнутую' фазу на основе серии изображений.
        Формулы соответствуют методическим указаниям (сдвиг 60 градусов).
        """
        # Убедимся, что изображения в float для вычислений
        images = [img.astype(np.float32) for img in images]
        
        I = images
        numerator = np.zeros_like(I[0])
        denominator = np.zeros_like(I[0])

        if steps == 3:
            numerator = 2 * I[0] - 3 * I[1] + I[2]
            denominator = np.sqrt(3) * (I[1] - I[2])
        elif steps == 4:
            numerator = 5 * (I[0] - I[1] - I[2] + I[3])
            denominator = np.sqrt(3) * (2 * I[0] + I[1] - I[2] - 2 * I[3])
            
        elif steps == 5:
            numerator = np.sqrt(3) * (2 * I[0] - 3 * I[1] - 4 * I[2] + 5 * I[4])
            denominator = 8 * I[0] + 3 * I[1] - 4 * I[2] - 6 * I[3] - I[4]
        else:
            raise ValueError("Поддерживаются только 3, 4, или 5 шагов.")     
        wrapped_phase = np.arctan2(numerator, denominator)
        return wrapped_phase

    def unwrap_phase(self, wrapped_phase):
        """Выполняет развёртку фазы."""
        # Используем готовую, быструю функцию из scikit-image
        return unwrap_phase(wrapped_phase)

    def scale_phase(self, phase_radians):
        return phase_radians * (self.lambda_angstrom / (2 * np.pi))

    def threshold_unwrap(self, height_map, threshold=0.8, iterations=1, horizontal=True, vertical=True):
        if height_map is None:
            return None
        h = height_map.copy()
        lam = self.lambda_angstrom
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

    def phase_jump(self, tile, threshold=0.8):
        lam = self.lambda_angstrom
        img = tile
        j = False
        h, w = img.shape
        for y in range(h):
            for x in range(1, w):
                if (img[y, x] - img[y, x - 1] > 0.5 * lam * threshold) or (img[y, x] - img[y, x - 1] < -0.5 * lam * threshold):
                    j = True
        for x in range(w):
            for y in range(1, h):
                if (img[y, x] - img[y - 1, x] > 0.5 * lam * threshold) or (img[y, x] - img[y - 1, x] < -0.5 * lam * threshold):
                    j = True
        return j

    def special_unwrap(self, p, tiles_mask, delimeter, threshold=0.8):
        lam = self.lambda_angstrom
        ResY, ResX = p.shape
        for i in range(3):
            for y in range(ResY):
                lastJ = p[y, 0]
                dh = 0.0
                for x in range(1, ResX):
                    J = p[y, x]
                    if not tiles_mask[x // delimeter, y // delimeter]:
                        if J - lastJ > 0.5 * lam * threshold:
                            dh -= 0.5 * lam
                        elif J - lastJ < -0.5 * lam * threshold:
                            dh += 0.5 * lam
                        lastJ = J
                    p[y, x] = J + dh
            for x in range(ResX):
                lastJ = p[0, x]
                dh = 0.0
                for y in range(1, ResY):
                    J = p[y, x]
                    if not tiles_mask[x // delimeter, y // delimeter]:
                        if J - lastJ > 0.5 * lam * threshold:
                            dh -= 0.5 * lam
                        elif J - lastJ < -0.5 * lam * threshold:
                            dh += 0.5 * lam
                        lastJ = J
                    p[y, x] = J + dh
        return p

    def tile_unwrap(self, height_map, delimeter=32, threshold=0.8, horizontal=True, vertical=True, use_special=True):
        img = height_map
        h, w = img.shape
        b = np.zeros_like(img)
        tiles_mask = np.zeros((max(1, w // delimeter), max(1, h // delimeter)), dtype=bool)
        for Y in range(0, h, delimeter):
            for X in range(0, w, delimeter):
                tile = img[Y:Y + delimeter, X:X + delimeter]
                unwrapped_tile = self.threshold_unwrap(tile, threshold=threshold, iterations=1, horizontal=horizontal, vertical=vertical)
                tiles_mask[X // delimeter, Y // delimeter] = self.phase_jump(unwrapped_tile, threshold=threshold)
                b[Y:Y + delimeter, X:X + delimeter] = unwrapped_tile
        if use_special:
            b = self.special_unwrap(b, tiles_mask, delimeter, threshold=threshold)
        return b

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
