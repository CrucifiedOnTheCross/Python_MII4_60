"""
DPI Recorder - модуль для записи последовательности фазовых измерений
Аналог функции DPI из Java версии
"""

import os
import time
import numpy as np
from datetime import datetime
import csv
from PySide6.QtCore import QObject, Signal


class DPIRecorder(QObject):
    """
    Класс для записи последовательности фазовых измерений (Digital Phase Interferometry)
    """
    
    # Сигналы для обновления GUI
    recording_started = Signal()
    recording_stopped = Signal()
    image_saved = Signal(int, str)  # номер изображения, путь к файлу
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.output_directory = ""
        self.image_count = 0
        self.start_time = None
        
    def start_recording(self, output_directory):
        """
        Начинает DPI запись
        
        Args:
            output_directory: путь к папке для сохранения файлов
        """
        if self.is_recording:
            return False
            
        # Создаем папку если она не существует
        try:
            os.makedirs(output_directory, exist_ok=True)
            self.output_directory = output_directory
            self.is_recording = True
            self.image_count = 0
            self.start_time = time.time()
            
            self.recording_started.emit()
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка создания папки: {str(e)}")
            return False
    
    def stop_recording(self):
        """
        Останавливает DPI запись
        """
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.recording_stopped.emit()
    
    def save_phase_data(self, phase_data, phase_image):
        """
        Сохраняет фазовые данные и изображение
        
        Args:
            phase_data: 2D numpy array с фазовыми данными
            phase_image: QImage или numpy array с изображением фазы
        """
        if not self.is_recording or phase_data is None:
            return
            
        try:
            # Генерируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"phase_{self.image_count:04d}_{timestamp}"
            
            # Сохраняем изображение
            image_path = os.path.join(self.output_directory, f"{base_filename}.png")
            if hasattr(phase_image, 'save'):
                # Если это QImage
                phase_image.save(image_path)
            else:
                # Если это numpy array
                import cv2
                cv2.imwrite(image_path, phase_image)
            
            # Сохраняем данные в CSV
            csv_path = os.path.join(self.output_directory, f"{base_filename}.csv")
            self._save_phase_to_csv(phase_data, csv_path)
            
            self.image_count += 1
            self.image_saved.emit(self.image_count, image_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка сохранения: {str(e)}")
    
    def _save_phase_to_csv(self, phase_data, csv_path):
        """
        Сохраняет фазовые данные в CSV файл
        
        Args:
            phase_data: 2D numpy array с фазовыми данными
            csv_path: путь к CSV файлу
        """
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            np.savetxt(csv_path, phase_data, fmt='%.6f', delimiter=',')
        except Exception as e:
            self.error_occurred.emit(f"Ошибка сохранения CSV: {str(e)}")
    
    def get_recording_info(self):
        """
        Возвращает информацию о текущей записи
        
        Returns:
            dict с информацией о записи
        """
        if not self.is_recording:
            return None
            
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'is_recording': self.is_recording,
            'output_directory': self.output_directory,
            'image_count': self.image_count,
            'elapsed_time': elapsed_time
        }
    
    def create_summary_file(self):
        """
        Создает файл с сводкой записи
        """
        if not self.output_directory:
            return
            
        try:
            summary_path = os.path.join(self.output_directory, "recording_summary.txt")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("DPI Recording Summary\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Recording Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output Directory: {self.output_directory}\n")
                f.write(f"Total Images: {self.image_count}\n")
                
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    f.write(f"Recording Duration: {elapsed:.2f} seconds\n")
                
                f.write(f"\nFiles saved:\n")
                for i in range(self.image_count):
                    f.write(f"  - phase_{i:04d}_*.png\n")
                    f.write(f"  - phase_{i:04d}_*.csv\n")
                    
        except Exception as e:
            self.error_occurred.emit(f"Ошибка создания сводки: {str(e)}")
