# MII4_60_Python/gui/main_window.py

import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QSlider, QCheckBox,
                               QMessageBox, QFileDialog)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, Slot

from hardware.controller import CameraController, ArduinoController
from core.phase_processor import PhaseProcessor
from core.visualizer import load_colormap, create_phase_image
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
                wrapped_phase = self.processor.compute_phase(images, self.params['steps'])
                unwrapped_phase = self.processor.unwrap_phase(wrapped_phase)
                
                if self.params['remove_trend']:
                    unwrapped_phase = self.processor.remove_linear_trend(unwrapped_phase)
                    
                final_image = create_phase_image(unwrapped_phase, self.colormap, self.params['inverse_color'])
                self.new_image.emit(final_image)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система Интерферометрического Анализа")
        self.setGeometry(100, 100, 1200, 700)

        # Инициализация контроллеров
        self.camera_ctrl = CameraController()
        self.arduino_ctrl = ArduinoController()
        
        self.measurement_thread = None
        self.is_measuring = False

        self._init_ui()
        self._populate_devices()

    def _init_ui(self):
        # Основной виджет и компоновка
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # --- Панель управления (слева) ---
        controls_layout = QVBoxLayout()
        
        # Камера
        controls_layout.addWidget(QLabel("Камера:"))
        self.cam_combo = QComboBox()
        self.cam_combo.currentTextChanged.connect(self.on_camera_change)
        controls_layout.addWidget(self.cam_combo)

        # Arduino
        controls_layout.addWidget(QLabel("Arduino (COM-порт):"))
        self.port_combo = QComboBox()
        self.port_combo.currentTextChanged.connect(self.on_port_change)
        controls_layout.addWidget(self.port_combo)
        
        # Параметры
        controls_layout.addWidget(QLabel(f"Количество шагов:"))
        self.steps_combo = QComboBox()
        self.steps_combo.addItems(["3", "4", "5"])
        controls_layout.addWidget(self.steps_combo)
        
        controls_layout.addWidget(QLabel(f"Задержка (мс): {config.DEFAULT_DELAY}"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(50, 2000)
        self.delay_slider.setValue(config.DEFAULT_DELAY)
        self.delay_slider.valueChanged.connect(lambda v: self.delay_slider.parent().findChild(QLabel, "Задержка (мс)").setText(f"Задержка (мс): {v}"))
        controls_layout.addWidget(self.delay_slider)
        
        # Чекбоксы
        self.inverse_cb = QCheckBox("Инвертировать цвета")
        self.trend_cb = QCheckBox("Удалить линейный тренд")
        controls_layout.addWidget(self.inverse_cb)
        controls_layout.addWidget(self.trend_cb)
        
        controls_layout.addStretch()
        
        # Кнопки
        self.start_btn = QPushButton("Старт/Стоп")
        self.start_btn.clicked.connect(self.toggle_measurement)
        controls_layout.addWidget(self.start_btn)

        self.save_btn = QPushButton("Сохранить изображение")
        self.save_btn.clicked.connect(self.save_image)
        controls_layout.addWidget(self.save_btn)
        
        # --- Область отображения (справа) ---
        display_layout = QVBoxLayout()
        self.image_label = QLabel("Изображение будет здесь")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        display_layout.addWidget(self.image_label)
        
        # Сборка
        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _populate_devices(self):
        self.cam_combo.addItems(self.camera_ctrl.list_cameras())
        self.port_combo.addItems(self.arduino_ctrl.list_ports())

    def on_camera_change(self, text):
        if "Нет" in text:
            self.camera_ctrl.stop()
            return
        cam_index = int(text.split()[-1])
        try:
            self.camera_ctrl.start(cam_index, config.DEFAULT_CAMERA_RESOLUTION)
        except IOError as e:
            QMessageBox.critical(self, "Ошибка камеры", str(e))

    def on_port_change(self, text):
        if "Нет" in text:
            self.arduino_ctrl.disconnect()
            return
        try:
            self.arduino_ctrl.connect(text)
        except IOError as e:
            QMessageBox.critical(self, "Ошибка Arduino", str(e))

    def toggle_measurement(self):
        if self.is_measuring:
            self.stop_measurement()
        else:
            self.start_measurement()

    def start_measurement(self):
        if not self.camera_ctrl.is_running:
            QMessageBox.warning(self, "Внимание", "Камера не выбрана или не работает.")
            return

        self.is_measuring = True
        self.start_btn.setText("Стоп")
        
        params = {
            'steps': int(self.steps_combo.currentText()),
            'delay': self.delay_slider.value(),
            'lambda': config.DEFAULT_LAMBDA,
            'inverse_color': self.inverse_cb.isChecked(),
            'remove_trend': self.trend_cb.isChecked(),
        }

        self.measurement_thread = MeasurementWorker(self.camera_ctrl, self.arduino_ctrl, params)
        self.measurement_thread.new_image.connect(self.update_image)
        self.measurement_thread.finished.connect(self.stop_measurement)
        self.measurement_thread.error.connect(self.on_measurement_error)
        self.measurement_thread.start()

    def stop_measurement(self):
        if self.measurement_thread:
            self.measurement_thread.stop()
        self.is_measuring = False
        self.start_btn.setText("Старт")

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_measurement_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка в процессе измерения", error_msg)
        self.stop_measurement()
        
    def save_image(self):
        pixmap = self.image_label.pixmap()
        if not pixmap:
            QMessageBox.warning(self, "Нечего сохранять", "Сначала проведите измерение.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG Images (*.png);;JPEG Images (*.jpg)")
        if path:
            pixmap.save(path)

    def closeEvent(self, event):
        self.camera_ctrl.stop()
        self.arduino_ctrl.disconnect()
        event.accept()