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
        # Векторизированная конвертация списка в массив сразу быстрее
        I = np.array(images, dtype=np.float32)
        
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
        return unwrap_phase(wrapped_phase)

    def scale_phase(self, phase_radians):
        return phase_radians * (self.lambda_angstrom / (2 * np.pi))

    def threshold_unwrap(self, height_map, threshold=0.8, iterations=1, horizontal=True, vertical=True):
        """Векторизированная версия пороговой развёртки."""
        if height_map is None:
            return None
        
        h = height_map.copy()
        lam = self.lambda_angstrom
        limit = 0.5 * lam * threshold
        correction_step = 0.5 * lam

        # Предварительно создаем массивы для индексов, если нужно, но здесь используем срезы
        for _ in range(max(1, iterations)):
            if horizontal:
                # Вычисляем разницу между соседними пикселями по горизонтали
                diff = np.diff(h, axis=1)
                
                # Создаем маску коррекций
                corrections = np.zeros_like(diff)
                corrections[diff > limit] = -correction_step
                corrections[diff < -limit] = correction_step
                
                # Накапливаем коррекции слева направо
                total_corrections = np.cumsum(corrections, axis=1)
                
                # Применяем к изображению (начиная со второго столбца)
                h[:, 1:] += total_corrections

            if vertical:
                # Вычисляем разницу между соседними пикселями по вертикали
                diff = np.diff(h, axis=0)
                
                # Создаем маску коррекций
                corrections = np.zeros_like(diff)
                corrections[diff > limit] = -correction_step
                corrections[diff < -limit] = correction_step
                
                # Накапливаем коррекции сверху вниз
                total_corrections = np.cumsum(corrections, axis=0)
                
                # Применяем к изображению (начиная со второй строки)
                h[1:, :] += total_corrections
                
        return h

    def phase_jump(self, tile, threshold=0.8):
        """Векторизированная проверка скачков фазы."""
        lam = self.lambda_angstrom
        limit = 0.5 * lam * threshold
        
        # Проверка по X и по Y сразу для всего тайла
        diff_x = np.diff(tile, axis=1)
        diff_y = np.diff(tile, axis=0)
        
        # Если хотя бы одна разница превышает предел
        if np.any(np.abs(diff_x) > limit) or np.any(np.abs(diff_y) > limit):
            return True
            
        return False

    def special_unwrap(self, p, tiles_mask, delimeter, threshold=0.8):
        """Векторизированная версия специальной развёртки с учетом маски плиток."""
        lam = self.lambda_angstrom
        limit = 0.5 * lam * threshold
        correction_step = 0.5 * lam
        
        rows, cols = p.shape
        
        # Создаем полную маску пикселей из маски тайлов
        # tiles_mask имеет размер (cols//del, rows//del)
        # Нам нужно растянуть её до (rows, cols)
        
        # ВАЖНО: tiles_mask индексируется как [x_block, y_block], то есть [col, row]
        # Нам нужно транспонировать её для repeat, чтобы получить (rows, cols) или правильно размножить
        
        # 1. Размножаем по оси X (столбцы)
        mask_expanded_x = np.repeat(tiles_mask, delimeter, axis=0) # (w, h//del)
        # 2. Размножаем по оси Y (строки)
        mask_expanded = np.repeat(mask_expanded_x, delimeter, axis=1) # (w, h)
        
        # Обрезаем или дополняем до точного размера изображения (если размер не кратен делителю)
        mask_full = np.zeros((cols, rows), dtype=bool)
        w_end = min(cols, mask_expanded.shape[0])
        h_end = min(rows, mask_expanded.shape[1])
        mask_full[:w_end, :h_end] = mask_expanded[:w_end, :h_end]
        
        # Транспонируем, чтобы получить (rows, cols) как у изображения p
        mask_full = mask_full.T 

        for _ in range(3):
            # --- Horizontal pass ---
            diff = np.diff(p, axis=1)
            # Маска для diff должна соответствовать целевому пикселю (x), от которого мы смотрим назад (x-1)
            # Если пиксель в "плохом" тайле (mask=True), мы НЕ применяем коррекцию (как в оригинале j-lastJ check skipped)
            valid_mask = ~mask_full[:, 1:]
            
            corrections = np.zeros_like(diff)
            # Коррекция применяется только там, где valid_mask is True
            # Условия для скачков
            jump_up = (diff > limit)
            jump_down = (diff < -limit)
            
            corrections[jump_up & valid_mask] = -correction_step
            corrections[jump_down & valid_mask] = correction_step
            
            p[:, 1:] += np.cumsum(corrections, axis=1)
            
            # --- Vertical pass ---
            diff = np.diff(p, axis=0)
            valid_mask = ~mask_full[1:, :]
            
            corrections = np.zeros_like(diff)
            jump_up = (diff > limit)
            jump_down = (diff < -limit)
            
            corrections[jump_up & valid_mask] = -correction_step
            corrections[jump_down & valid_mask] = correction_step
            
            p[1:, :] += np.cumsum(corrections, axis=0)
            
        return p

    def tile_unwrap(self, height_map, delimeter=32, threshold=0.8, horizontal=True, vertical=True, use_special=True):
        img = height_map
        h, w = img.shape
        b = np.zeros_like(img)
        
        # Вычисляем размеры сетки тайлов
        nx = max(1, w // delimeter)
        ny = max(1, h // delimeter)
        tiles_mask = np.zeros((nx, ny), dtype=bool)
        
        # Этот цикл сложно полностью векторизовать, так как tiles_mask зависит от результата threshold_unwrap для каждого тайла.
        # Но сам threshold_unwrap теперь быстрый, так что цикл по тайлам (которых немного) не будет узким местом.
        for Y_idx in range(ny):
            for X_idx in range(nx):
                Y = Y_idx * delimeter
                X = X_idx * delimeter
                
                # Обработка краев изображения, если размер не кратен delimeter
                Y_end = min(Y + delimeter, h)
                X_end = min(X + delimeter, w)
                
                tile = img[Y:Y_end, X:X_end]
                
                # Вызываем быструю версию
                unwrapped_tile = self.threshold_unwrap(tile, threshold=threshold, iterations=1, horizontal=horizontal, vertical=vertical)
                
                # Проверка скачков
                has_jump = self.phase_jump(unwrapped_tile, threshold=threshold)
                tiles_mask[X_idx, Y_idx] = has_jump
                
                b[Y:Y_end, X:X_end] = unwrapped_tile
                
        if use_special:
            b = self.special_unwrap(b, tiles_mask, delimeter, threshold=threshold)
            
        return b

    def remove_linear_trend(self, phase_data):
        """
        Удаляет линейный тренд (наклон), рассчитанный по краям изображения,
        чтобы соответствовать поведению оригинального Java кода.
        """
        if phase_data is None or phase_data.size == 0:
            return phase_data
            
        # Работаем с копией данных
        result = phase_data.copy()
        rows, cols = result.shape

        # 1. Горизонтальный тренд (расчет по первой строке)
        # В Java: horizontal[x] = k[x][0] -> linReg -> вычитание из всех строк
        x = np.arange(cols)
        first_row = result[0, :] # Берем профиль первой строки
        
        # Аппроксимируем прямой линией (степень 1)
        coeffs_h = np.polyfit(x, first_row, 1) 
        trend_h = np.polyval(coeffs_h, x)
        
        # Вычитаем полученный тренд из всех строк изображения
        # Использование broadcasting [None, :] применяет вектор (cols,) ко всей матрице (rows, cols)
        result -= trend_h[np.newaxis, :]
        
        # 2. Вертикальный тренд (расчет по первому столбцу)
        # В Java: vertical[y] = k[0][y] -> linReg -> вычитание из всех столбцов
        y = np.arange(rows)
        first_col = result[:, 0] # Берем профиль первого столбца (после удаления горизонтального тренда)
        
        coeffs_v = np.polyfit(y, first_col, 1)
        trend_v = np.polyval(coeffs_v, y)
        
        # Вычитаем полученный тренд из всех столбцов
        # Использование broadcasting [:, None] превращает вектор в столбец (rows, 1)
        result -= trend_v[:, np.newaxis]
        
        return result

    def remove_polynomial_trend(self, phase_data):
        """
        Удаляет полиномиальный тренд из фазовых данных (аналог sphereTrend из Java версии)
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
        
        # Удаляем горизонтальный тренд (векторизировано)
        result -= horizontal_trend[np.newaxis, :]
            
        # Удаляем вертикальный тренд (векторизировано)
        result -= vertical_trend[:, np.newaxis]
            
        return result
    
    def _fit_polynomial_trend(self, profile, size):
        """
        Вычисляет полиномиальный тренд второго порядка для одномерного профиля
        """
        # Создаем матрицу A для полинома второго порядка: y = w1 + w2*x + w3*x^2
        x = np.arange(size)
        # Используем встроенный polyfit, он быстрее и надежнее ручного обращения матриц
        try:
            coeffs = np.polyfit(x, profile, 2) # Возвращает [a, b, c] для ax^2 + bx + c
            trend = np.polyval(coeffs, x)
            return trend
        except Exception:
            return np.zeros(size)