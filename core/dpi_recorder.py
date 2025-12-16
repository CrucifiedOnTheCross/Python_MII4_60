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
import threading
import queue


class DPIRecorder(QObject):
    """
    Класс для записи последовательности фазовых измерений (Digital Phase Interferometry)
    """
    
    # Сигналы для обновления GUI
    recording_started = Signal()
    recording_stopped = Signal()
    image_saved = Signal(int, str)  # номер изображения, путь к файлу
    error_occurred = Signal(str)
    values_ready = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.output_directory = ""
        self.image_count = 0
        self.start_time = None
        self.params = {}
        self._queue = queue.Queue()
        self._writer_thread = None
        self._stop_event = threading.Event()
        self._writing = False
        self._experiment_finished = False
        
    def start_recording(self, output_directory, params=None):
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
            self.params = params or {}
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            self._stop_event.clear()
            self._experiment_finished = False
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()
            
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
        self._stop_event.set()
        t = self._writer_thread
        if t is not None:
            t.join(timeout=10)
        # Значение записывается после завершения эксперимента либо здесь, если запись остановлена вручную
        self.create_values_file()
        self.recording_stopped.emit()
    
    def save_phase_data(self, phase_data, phase_image=None):
        if not self.is_recording or phase_data is None:
            return
        try:
            self._queue.put_nowait(phase_data)
        except Exception as e:
            self.error_occurred.emit(f"Ошибка постановки в очередь: {str(e)}")
    
    def _save_phase_to_csv(self, phase_data, csv_path):
        """
        Сохраняет фазовые данные в CSV файл
        
        Args:
            phase_data: 2D numpy array с фазовыми данными
            csv_path: путь к CSV файлу
        """
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            data_int = np.rint(phase_data).astype(np.int64)
            np.savetxt(csv_path, data_int, fmt='%d', delimiter=',')
        except Exception as e:
            self.error_occurred.emit(f"Ошибка сохранения CSV: {str(e)}")
    
    def _writer_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                phase_data = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._writing = True
                num = self.image_count + 1
                base_filename = f"test{num}"
                csv_path = os.path.join(self.output_directory, f"{base_filename}.csv")
                self._save_phase_to_csv(phase_data, csv_path)
                self.image_count = num
                self.image_saved.emit(self.image_count, csv_path)
            except Exception as e:
                self.error_occurred.emit(f"Ошибка записи: {str(e)}")
            finally:
                self._writing = False
                self._queue.task_done()
            # Если эксперимент завершён и очередь пуста - пишем values
            if self._experiment_finished and self._queue.empty():
                try:
                    self.create_values_file()
                    self.values_ready.emit(os.path.join(self.output_directory, "values.txt"))
                except Exception as e:
                    self.error_occurred.emit(f"Ошибка финализации values: {str(e)}")
                # Сбрасываем флаг, чтобы не создавать файл повторно
                self._experiment_finished = False
    
    def wait_until_idle(self, timeout=10.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._queue.empty() and not self._writing:
                return True
            time.sleep(0.05)
        return False
    
    def create_values_file_after_flush(self):
        self.wait_until_idle(timeout=10.0)
        self.create_values_file()

    def mark_experiment_end(self):
        self._experiment_finished = True
    
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
    
    def create_values_file(self):
        if not self.output_directory:
            return
            
        try:
            values_path = os.path.join(self.output_directory, "values.txt")
            elapsed_ms = int((time.time() - self.start_time) * 1000) if self.start_time else 0
            steps = int(self.params.get('steps', 0))
            wavelength = float(self.params.get('lambda_angstrom', 0.0))
            threshold = float(self.params.get('threshold', 0.0))
            delay = int(self.params.get('delay', 0))
            with open(values_path, 'w', encoding='utf-8') as f:
                f.write(f"Time ms:  {elapsed_ms}\n")
                f.write(f"Quantity:  {self.image_count}\n")
                f.write(f"Algorythm:  {steps} step\n")
                f.write(f"Wavelagth:  {wavelength:.1f}\n")
                f.write(f"Threshold:  {threshold}\n")
                f.write(f"Delay:  {delay}\n")
        except Exception as e:
            self.error_occurred.emit(f"Ошибка создания values: {str(e)}")
