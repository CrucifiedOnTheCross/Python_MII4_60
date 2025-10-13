# MII4_60_Python/gui/main_window.py

import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QSlider, QCheckBox,
                               QMessageBox, QFileDialog, QGroupBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, Slot

from hardware.controller import CameraController, ArduinoController
from core.phase_processor import PhaseProcessor
from core.visualizer import load_colormap, create_phase_image
from core.dpi_recorder import DPIRecorder
from core.visualizer import create_interferogram, save_data_to_csv
import config
import time

# Worker для выполнения захвата в отдельном потоке
class MeasurementWorker(QThread):
    new_image = Signal(np.ndarray)
    finished = Signal()
    error = Signal(str)

    def __init__(self, camera, arduino, params):
        super().__init__()
        self.camera = camera
        self.arduino = arduino
        self.params = params
        self.is_running = True
        self.processor = PhaseProcessor(lambda_nm=params['lambda'])
        self.colormap = load_colormap(config.COLORMAP_FILE)

    def run(self):
        try:
            images = []
            for i in range(self.params['steps']):
                if not self.is_running:
                    break
                
                # Команда Arduino, если он подключен
                if self.arduino and self.arduino.is_connected:
                    self.arduino.send_step_command(i)
                
                time.sleep(self.params['delay'] / 1000.0)
                
                frame = self.camera.get_frame()
                if frame is None:
                    raise Exception("Не удалось получить кадр с камеры.")
                images.append(frame)
            
            if len(images) == self.params['steps']:
                # Вычисляем фазу
                phase_data = self.processor.compute_phase(images)
                
                # Применяем развертку фазы, если включена
                if self.params.get('unwrap', False):
                    phase_data = self.processor.unwrap_phase(phase_data)
                
                # Удаляем тренд, если включено
                if self.params.get('remove_trend', False):
                    phase_data = self.processor.remove_linear_trend(phase_data)
                
                # Создаем изображение
                phase_image = create_phase_image(
                    phase_data, 
                    self.colormap, 
                    inverse=self.params.get('inverse', False),
                    rainbow=self.params.get('rainbow', False)
                )
                
                self.new_image.emit(phase_image)
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MII4_60 Python - Интерферометр")
        self.setGeometry(100, 100, 1200, 800)
        
        # Инициализация контроллеров
        self.camera_ctrl = CameraController()
        self.arduino_ctrl = ArduinoController()
        
        # Инициализация DPI recorder
        self.dpi_recorder = DPIRecorder()
        self.dpi_recorder.recording_started.connect(self.on_dpi_recording_started)
        self.dpi_recorder.recording_stopped.connect(self.on_dpi_recording_stopped)
        self.dpi_recorder.image_saved.connect(self.on_dpi_image_saved)
        self.dpi_recorder.error_occurred.connect(self.on_error)
        
        # Список изображений для интерферограммы
        self.captured_images = []
        self.current_phase_data = None
        
        # Worker для измерений
        self.worker = None
        
        self._init_ui()
        self._populate_devices()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель - элементы управления
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(300)
        self.controls_layout = QVBoxLayout(controls_widget)
        
        # Выбор камеры
        self.controls_layout.addWidget(QLabel("Камера:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self.on_camera_change)
        self.controls_layout.addWidget(self.camera_combo)
        
        # Выбор порта Arduino
        self.controls_layout.addWidget(QLabel("Arduino порт:"))
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(self.on_port_change)
        self.controls_layout.addWidget(self.port_combo)
        
        # Количество шагов
        self.controls_layout.addWidget(QLabel("Количество шагов:"))
        self.steps_combo = QComboBox()
        self.steps_combo.addItems(["3", "4", "5"])
        self.steps_combo.setCurrentText("4")
        self.controls_layout.addWidget(self.steps_combo)
        
        # Длина волны
        self.controls_layout.addWidget(QLabel("Длина волны (нм):"))
        self.lambda_combo = QComboBox()
        self.lambda_combo.addItems(["632.8", "543.5", "488.0"])
        self.lambda_combo.setCurrentText("632.8")
        self.controls_layout.addWidget(self.lambda_combo)
        
        # Задержка
        self.controls_layout.addWidget(QLabel("Задержка (мс):"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(10, 1000)
        self.delay_slider.setValue(100)
        self.delay_label = QLabel("100")
        self.delay_slider.valueChanged.connect(lambda v: self.delay_label.setText(str(v)))
        self.controls_layout.addWidget(self.delay_slider)
        self.controls_layout.addWidget(self.delay_label)
        
        # Чекбоксы
        self.unwrap_checkbox = QCheckBox("Развертка фазы")
        self.controls_layout.addWidget(self.unwrap_checkbox)
        
        self.trend_checkbox = QCheckBox("Удаление тренда")
        self.controls_layout.addWidget(self.trend_checkbox)
        
        self.inverse_checkbox = QCheckBox("Инверсия цветов")
        self.controls_layout.addWidget(self.inverse_checkbox)
        
        self.rainbow_checkbox = QCheckBox("Радужная палитра")
        self.controls_layout.addWidget(self.rainbow_checkbox)
        
        # Добавляем расширенные элементы управления
        self.setup_advanced_controls()
        
        # Кнопки
        self.start_button = QPushButton("Начать измерение")
        self.start_button.clicked.connect(self.toggle_measurement)
        self.controls_layout.addWidget(self.start_button)
        
        self.save_button = QPushButton("Сохранить изображение")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        self.controls_layout.addWidget(self.save_button)
        
        self.controls_layout.addStretch()
        
        # Правая панель - отображение изображения
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid gray")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Изображение будет отображено здесь")
        
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.image_label)

    def setup_advanced_controls(self):
        """Настройка дополнительных элементов управления"""
        
        # Группа для полиномиального тренда
        trend_group = QGroupBox("Удаление тренда")
        trend_layout = QVBoxLayout()
        
        self.polynomial_trend_checkbox = QCheckBox("Полиномиальный тренд")
        self.polynomial_trend_checkbox.setToolTip("Удаление полиномиального тренда (sphereTrend)")
        trend_layout.addWidget(self.polynomial_trend_checkbox)
        
        trend_group.setLayout(trend_layout)
        
        # Группа для DPI записи
        dpi_group = QGroupBox("DPI Запись")
        dpi_layout = QVBoxLayout()
        
        self.dpi_record_button = QPushButton("Начать DPI запись")
        self.dpi_record_button.clicked.connect(self.toggle_dpi_recording)
        dpi_layout.addWidget(self.dpi_record_button)
        
        self.dpi_status_label = QLabel("DPI: Остановлена")
        dpi_layout.addWidget(self.dpi_status_label)
        
        dpi_group.setLayout(dpi_layout)
        
        # Группа для интерферограмм
        interferogram_group = QGroupBox("Интерферограммы")
        interferogram_layout = QVBoxLayout()
        
        self.save_interferogram_checkbox = QCheckBox("Сохранять интерферограммы")
        self.save_interferogram_checkbox.setToolTip("Сохранение интерферограмм вместо фазовых изображений")
        interferogram_layout.addWidget(self.save_interferogram_checkbox)
        
        interferogram_group.setLayout(interferogram_layout)
        
        # Группа для экспорта данных
        export_group = QGroupBox("Экспорт данных")
        export_layout = QVBoxLayout()
        
        self.export_csv_button = QPushButton("Экспорт в CSV")
        self.export_csv_button.clicked.connect(self.export_phase_data_csv)
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)
        
        export_group.setLayout(export_layout)
        
        # Добавляем группы в основной layout
        self.controls_layout.addWidget(trend_group)
        self.controls_layout.addWidget(dpi_group)
        self.controls_layout.addWidget(interferogram_group)
        self.controls_layout.addWidget(export_group)

    def _populate_devices(self):
        """Заполняет списки доступных устройств."""
        # Заполняем список камер
        cameras = self.camera_ctrl.list_cameras()
        self.camera_combo.clear()
        self.camera_combo.addItems(cameras)
        
        # Заполняем список портов Arduino
        ports = self.arduino_ctrl.list_ports()
        self.port_combo.clear()
        self.port_combo.addItems(ports)

    def on_camera_change(self, text):
        if text:
            try:
                camera_id = int(text.split()[-1])
                if self.camera_ctrl.connect(camera_id):
                    QMessageBox.information(self, "Успех", f"Камера {camera_id} подключена")
                else:
                    QMessageBox.warning(self, "Ошибка", f"Не удалось подключить камеру {camera_id}")
            except:
                pass

    def on_port_change(self, text):
        if text and text != "Нет доступных портов":
            try:
                if self.arduino_ctrl.connect(text):
                    QMessageBox.information(self, "Успех", f"Arduino подключен к {text}")
                else:
                    QMessageBox.warning(self, "Ошибка", f"Не удалось подключить Arduino к {text}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка подключения: {str(e)}")

    def toggle_measurement(self):
        if self.worker is None or not self.worker.isRunning():
            self.start_measurement()
        else:
            self.stop_measurement()

    def start_measurement(self):
        if not self.camera_ctrl.is_connected:
            QMessageBox.warning(self, "Предупреждение", "Камера не подключена")
            return
        
        params = {
            'steps': int(self.steps_combo.currentText()),
            'lambda': float(self.lambda_combo.currentText()),
            'delay': self.delay_slider.value(),
            'unwrap': self.unwrap_checkbox.isChecked(),
            'remove_trend': self.trend_checkbox.isChecked(),
            'inverse': self.inverse_checkbox.isChecked(),
            'rainbow': self.rainbow_checkbox.isChecked()
        }
        
        self.worker = MeasurementWorker(self.camera_ctrl, self.arduino_ctrl, params)
        self.worker.new_image.connect(self.update_image)
        self.worker.finished.connect(self.on_measurement_finished)
        self.worker.error.connect(self.on_measurement_error)
        self.worker.start()
        
        self.start_button.setText("Остановить измерение")

    def stop_measurement(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.start_button.setText("Начать измерение")

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        # Применяем полиномиальное удаление тренда если включено
        if hasattr(self, 'polynomial_trend_checkbox') and self.polynomial_trend_checkbox.isChecked():
            # Получаем фазовые данные из изображения (это упрощение)
            # В реальности нужно передавать фазовые данные отдельно
            pass
        
        # Сохраняем текущие данные для экспорта
        self.export_csv_button.setEnabled(True)
        
        # Если включена DPI запись, сохраняем данные
        if self.dpi_recorder.is_recording:
            # Конвертируем numpy array в QImage для сохранения
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.dpi_recorder.save_phase_data(cv_img, q_image)
        
        # Отображаем изображение
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Масштабируем изображение для отображения
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.save_button.setEnabled(True)

    def on_measurement_finished(self):
        self.start_button.setText("Начать измерение")

    def on_measurement_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка измерения", error_msg)
        self.start_button.setText("Начать измерение")

    def save_image(self):
        if self.image_label.pixmap():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Сохранить изображение", "phase_image.png", 
                "PNG файлы (*.png);;JPEG файлы (*.jpg)"
            )
            if filename:
                self.image_label.pixmap().save(filename)
                QMessageBox.information(self, "Успех", f"Изображение сохранено: {filename}")

    def toggle_dpi_recording(self):
        """Переключение DPI записи"""
        if not self.dpi_recorder.is_recording:
            # Выбираем папку для сохранения
            folder = QFileDialog.getExistingDirectory(
                self, 
                "Выберите папку для DPI записи",
                ""
            )
            if folder:
                if self.dpi_recorder.start_recording(folder):
                    self.dpi_record_button.setText("Остановить DPI запись")
                    self.dpi_status_label.setText("DPI: Запись...")
        else:
            self.dpi_recorder.stop_recording()
            self.dpi_record_button.setText("Начать DPI запись")
            self.dpi_status_label.setText("DPI: Остановлена")
    
    def on_dpi_recording_started(self):
        """Обработчик начала DPI записи"""
        self.dpi_record_button.setText("Остановить DPI запись")
        self.dpi_status_label.setText("DPI: Запись...")
    
    def on_dpi_recording_stopped(self):
        """Обработчик остановки DPI записи"""
        self.dpi_record_button.setText("Начать DPI запись")
        self.dpi_status_label.setText("DPI: Остановлена")
        # Создаем файл сводки
        self.dpi_recorder.create_summary_file()
    
    def on_dpi_image_saved(self, image_number, filepath):
        """Обработчик сохранения изображения DPI"""
        self.dpi_status_label.setText(f"DPI: Сохранено {image_number} изображений")
    
    def on_error(self, error_msg):
        """Обработчик ошибок"""
        QMessageBox.critical(self, "Ошибка", error_msg)
    
    def export_phase_data_csv(self):
        """Экспорт фазовых данных в CSV"""
        if not hasattr(self, 'current_phase_data') or self.current_phase_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить фазовые данные",
            "phase_data.csv",
            "CSV файлы (*.csv)"
        )
        
        if filename:
            if save_data_to_csv(self.current_phase_data, filename):
                QMessageBox.information(self, "Успех", f"Данные сохранены в {filename}")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить данные")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        self.camera_ctrl.stop()
        self.arduino_ctrl.disconnect()
        event.accept()