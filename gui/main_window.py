# MII4_60_Python/gui/main_window.py

import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QSlider, QCheckBox,
                               QMessageBox, QFileDialog, QGroupBox, QLineEdit, QSpinBox,
                               QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QSizePolicy, QFrame, QScrollArea, QDialog)
from PySide6.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator, QFont
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer

from hardware.controller import CameraController, ArduinoController
from core.phase_processor import PhaseProcessor
from core.visualizer import load_colormap, create_phase_image_gray, create_phase_image
from core.dpi_recorder import DPIRecorder
from core.visualizer import create_interferogram, save_image_as_csv
import config
import time

# Worker для постоянной трансляции видео с камеры
class CameraStreamWorker(QThread):
    new_frame = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, camera_controller):
        super().__init__()
        self.camera_controller = camera_controller
        self.is_running = False

    def run(self):
        self.is_running = True
        while self.is_running:
            try:
                if self.camera_controller.is_running:
                    frame = self.camera_controller.get_frame()
                    if frame is not None:
                        # Конвертируем в цветное изображение для отображения
                        if len(frame.shape) == 2:  # Если серое изображение
                            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        else:
                            frame_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.new_frame.emit(frame_color)
                self.msleep(33)  # ~30 FPS
            except Exception as e:
                self.error.emit(f"Ошибка захвата кадра: {str(e)}")
                break

    def stop(self):
        self.is_running = False
        self.wait()

# Worker для выполнения захвата в отдельном потоке
class MeasurementWorker(QThread):
    new_phase_image = Signal(np.ndarray)
    new_interferogram = Signal(np.ndarray)
    phase_data_ready = Signal(np.ndarray)
    new_step_image = Signal(np.ndarray)
    finished = Signal()
    error = Signal(str)

    def __init__(self, camera, arduino, params):
        super().__init__()
        self.camera = camera
        self.arduino = arduino
        self.params = params
        self.is_running = True
        self.processor = PhaseProcessor(lambda_angstrom=params['lambda_angstrom'])
        self.colormap = load_colormap(config.COLORMAP_FILE)

    def run(self):
        try:
            series_count = int(self.params.get('series_count', 64))
            for _ in range(series_count):
                if not self.is_running:
                    break
                images = []
                for i in range(self.params['steps']):
                    if not self.is_running:
                        break
                    if self.arduino and self.arduino.is_connected:
                        self.arduino.send_step_command(i)
                    time.sleep(self.params['delay'] / 1000.0)
                    frame = self.camera.get_frame()
                    if frame is None:
                        raise Exception("Не удалось получить кадр с камеры.")
                    images.append(frame)
                    self.new_step_image.emit(frame)

                if len(images) == self.params['steps']:
                    phase_wrapped = self.processor.compute_phase(images, self.params['steps'])
                    use_scale = self.params.get('scale', True)
                    threshold = self.params.get('threshold', 0.8)
                    if self.params.get('unwrap', False):
                        if use_scale:
                            height_map = self.processor.scale_phase(phase_wrapped)
                            if self.params.get('tile_unwrap', False):
                                height_map = self.processor.tile_unwrap(height_map, delimeter=self.params.get('tile_size', 32), threshold=threshold, horizontal=True, vertical=True, use_special=True)
                            else:
                                height_map = self.processor.threshold_unwrap(height_map, threshold=threshold, iterations=2, horizontal=True, vertical=True)
                            phase_data = height_map
                        else:
                            unwrapped = self.processor.unwrap_phase(phase_wrapped)
                            phase_data = unwrapped
                    else:
                        phase_data = self.processor.scale_phase(phase_wrapped) if use_scale else phase_wrapped
                    if self.params.get('remove_trend', False):
                        phase_data = self.processor.remove_linear_trend(phase_data)
                    if self.params.get('rainbow', False):
                        phase_image = create_phase_image(
                            phase_data,
                            self.colormap,
                            inverse=self.params.get('inverse', False)
                        )
                    else:
                        phase_image = create_phase_image_gray(
                            phase_data,
                            inverse=self.params.get('inverse', False)
                        )
                    self.phase_data_ready.emit(phase_data)
                    self.new_phase_image.emit(phase_image)

                    interferogram = create_interferogram(images, 'average')
                    if interferogram is not None:
                        if len(interferogram.shape) == 2:
                            interferogram = cv2.cvtColor(interferogram, cv2.COLOR_GRAY2RGB)
                        else:
                            interferogram = cv2.cvtColor(interferogram, cv2.COLOR_BGR2RGB)
                        self.new_interferogram.emit(interferogram)
            
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
        
        # Worker для постоянной трансляции видео
        self.camera_stream_worker = None
        self.main_view_mode = 'camera'
        
        self._init_ui()
        self._populate_devices()
        self.last_phase_qimage = None

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель - элементы управления
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(360)
        controls_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
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
        self.controls_layout.addWidget(QLabel("Длина волны (Å):"))
        self.lambda_input = QLineEdit()
        self.lambda_input.setText("7500")
        self.lambda_input.setValidator(QDoubleValidator(4000.0, 8000.0, 1))
        self.controls_layout.addWidget(self.lambda_input)
        
        # Задержка
        self.controls_layout.addWidget(QLabel("Задержка (мс):"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(10, 1000)
        self.delay_slider.setValue(200)
        self.delay_label = QLabel("200")
        self.delay_slider.valueChanged.connect(lambda v: self.delay_label.setText(str(v)))
        self.controls_layout.addWidget(self.delay_slider)
        self.controls_layout.addWidget(self.delay_label)
        
        # Количество фаз (серий)
        self.controls_layout.addWidget(QLabel("Количество фаз:"))
        self.series_spin = QSpinBox()
        self.series_spin.setRange(1, 1000)
        self.series_spin.setValue(64)
        self.controls_layout.addWidget(self.series_spin)
        
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
        
        # Кнопка для включения/выключения трансляции камеры
        self.stream_button = QPushButton("Включить трансляцию")
        self.stream_button.clicked.connect(self.toggle_camera_stream)
        self.controls_layout.addWidget(self.stream_button)

        self.toggle_view_button = QPushButton("Показать фазу")
        self.toggle_view_button.clicked.connect(self.toggle_main_view)
        self.controls_layout.addWidget(self.toggle_view_button)


        self.save_settings_button = QPushButton("Сохранить настройки")
        self.save_settings_button.clicked.connect(self.save_settings)
        self.controls_layout.addWidget(self.save_settings_button)

        self.load_settings_button = QPushButton("Загрузить настройки")
        self.load_settings_button.clicked.connect(self.load_settings)
        self.controls_layout.addWidget(self.load_settings_button)
        
        self.controls_layout.addStretch()

        self.phase_view_small = GraphicsImageView()
        self.phase_view_small.setToolTip("Фаза (ч/б)")
        self.phase_view_small.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.phase_view_small.setFixedHeight(220)
        self.controls_layout.addWidget(self.phase_view_small)
        
        right_layout = QHBoxLayout()
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.interferogram_view = GraphicsImageView()
        self.interferogram_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.interferogram_view.setToolTip("Интерферограмма")
        interfer_box = QVBoxLayout()
        interfer_box.setContentsMargins(0, 0, 0, 0)
        interfer_box.setSpacing(0)
        interfer_box.addWidget(self.interferogram_view)
        interfer_widget = QWidget()
        interfer_widget.setLayout(interfer_box)

        # Нижняя полоска миниатюр удалена

        right_layout.addWidget(interfer_widget, 1)

        container = QWidget()
        container.setLayout(right_layout)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_layout.addWidget(container, 1)
        main_layout.addWidget(controls_scroll)

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
        dpi_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        dpi_group.setMinimumHeight(110)
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
        
        self.export_csv_button = QPushButton("Сохранить фазовое изображение")
        self.export_csv_button.clicked.connect(self.export_phase_data_csv)
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)
        
        export_group.setLayout(export_layout)
        
        # Добавляем группы в основной layout
        self.controls_layout.addWidget(trend_group)
        self.controls_layout.addWidget(dpi_group)
        self.controls_layout.addWidget(interferogram_group)
        self.controls_layout.addWidget(export_group)

        threshold_group = QGroupBox("Порог развёртки")
        threshold_layout = QVBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(80)
        self.threshold_label = QLabel("0.80")
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_label.setText(f"{v/100:.2f}"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        threshold_group.setLayout(threshold_layout)
        self.controls_layout.addWidget(threshold_group)

        tile_group = QGroupBox("Плиточная развёртка")
        tile_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        tile_group.setMinimumHeight(140)
        tile_layout = QVBoxLayout()
        self.tile_unwrap_checkbox = QCheckBox("Включить плиточную развёртку")
        tile_layout.addWidget(self.tile_unwrap_checkbox)
        self.tile_size_slider = QSlider(Qt.Horizontal)
        self.tile_size_slider.setRange(8, 128)
        self.tile_size_slider.setValue(32)
        self.tile_size_label = QLabel("32")
        self.tile_size_slider.valueChanged.connect(lambda v: self.tile_size_label.setText(str(v)))
        tile_layout.addWidget(self.tile_size_slider)
        tile_layout.addWidget(self.tile_size_label)
        tile_group.setLayout(tile_layout)
        self.controls_layout.addWidget(tile_group)

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

    def _refresh_devices(self):
        """Обновляет список устройств, предварительно освобождая ресурсы Arduino."""
        try:
            if self.arduino_ctrl.is_connected:
                self.arduino_ctrl.disconnect()
                time.sleep(0.2)
        except Exception:
            pass
        self._populate_devices()

    def on_camera_change(self, text):
        """Обработчик изменения выбранной камеры."""
        if text and text != "Нет доступных камер":
            try:
                # Останавливаем текущую трансляцию если она активна
                if self.camera_stream_worker and self.camera_stream_worker.isRunning():
                    self.stop_camera_stream()
                
                # Останавливаем текущую камеру
                self.camera_ctrl.stop()
                
                # Извлекаем индекс камеры из текста
                camera_index = int(text.split()[-1])
                self.camera_ctrl.start(camera_index)
                print(f"Переключились на камеру {camera_index}")
                
                # Автоматически запускаем трансляцию если камера успешно подключена
                if self.camera_ctrl.is_running:
                    self.start_camera_stream()
                    
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось подключиться к камере: {e}")

    def on_port_change(self, text):
        if text and text != "Нет доступных портов":
            try:
                if "Занят" in text:
                    QMessageBox.warning(self, "Порт занят", "Выбранный COM-порт занят другой программой. Закройте сторонние приложения или выберите свободный порт.")
                    return
                device = text.split()[0]
                ok = self.arduino_ctrl.connect(device)
                if ok:
                    QMessageBox.information(self, "Успех", f"Arduino подключен к {text}")
                else:
                    auto = self.arduino_ctrl.connect_auto()
                    if auto:
                        QMessageBox.information(self, "Успех", "Arduino подключен автоматически")
                    else:
                        QMessageBox.warning(self, "Ошибка", f"Не удалось подключить Arduino к {text}")
            except Exception as e:
                auto = self.arduino_ctrl.connect_auto()
                if auto:
                    QMessageBox.information(self, "Успех", "Arduino подключен автоматически")
                else:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка подключения: {str(e)}")

    def test_arduino(self):
        text = self.port_combo.currentText()
        if not text or text == "Нет доступных портов":
            QMessageBox.warning(self, "Предупреждение", "Нет доступного порта Arduino")
            return
        try:
            if "Занят" in text:
                QMessageBox.warning(self, "Порт занят", "Выбранный COM-порт занят другой программой. Закройте сторонние приложения или выберите свободный порт.")
                return
            device = text.split()[0]
            if not self.arduino_ctrl.is_connected:
                ok = self.arduino_ctrl.connect(device)
                if not ok:
                    if not self.arduino_ctrl.connect_auto():
                        QMessageBox.warning(self, "Ошибка", f"Не удалось подключиться к {text}")
                        return
            self.arduino_ctrl.send_step_command(0)
            self.arduino_ctrl.reset_all_pins()
            QMessageBox.information(self, "Успех", "Команда отправлена на Arduino")
        except Exception as e:
            if self.arduino_ctrl.connect_auto():
                self.arduino_ctrl.send_step_command(0)
                self.arduino_ctrl.reset_all_pins()
                QMessageBox.information(self, "Успех", "Команда отправлена на Arduino")
            else:
                QMessageBox.critical(self, "Ошибка", f"Тест не выполнен: {str(e)}")

    def blink_arduino(self):
        text = self.port_combo.currentText()
        if not text or text == "Нет доступных портов":
            QMessageBox.warning(self, "Предупреждение", "Нет доступного порта Arduino")
            return
        try:
            if "Занят" in text:
                QMessageBox.warning(self, "Порт занят", "Выбранный COM-порт занят другой программой. Закройте сторонние приложения или выберите свободный порт.")
                return
            device = text.split()[0]
            if not self.arduino_ctrl.is_connected:
                ok = self.arduino_ctrl.connect(device)
                if not ok:
                    if not self.arduino_ctrl.connect_auto():
                        QMessageBox.warning(self, "Ошибка", f"Не удалось подключиться к {text}")
                        return
            period = self.delay_slider.value()
            success = True
            for _ in range(5):
                if not self.arduino_ctrl.blink(duration_ms=period):
                    success = False
                    break
                time.sleep(period / 1000.0)
            if success:
                QMessageBox.information(self, "Успех", "Arduino мигнул 5 раз")
            else:
                QMessageBox.warning(self, "Предупреждение", "Команда мигания не выполнена")
        except Exception as e:
            if self.arduino_ctrl.connect_auto():
                period = self.delay_slider.value()
                success = True
                for _ in range(5):
                    if not self.arduino_ctrl.blink(duration_ms=period):
                        success = False
                        break
                    time.sleep(period / 1000.0)
                if success:
                    QMessageBox.information(self, "Успех", "Arduino мигнул 5 раз")
                else:
                    QMessageBox.warning(self, "Предупреждение", "Команда мигания не выполнена")
            else:
                QMessageBox.critical(self, "Ошибка", f"Мигание не выполнено: {str(e)}")

    def toggle_measurement(self):
        if self.worker is None or not self.worker.isRunning():
            self.start_measurement()
        else:
            self.stop_measurement()

    def start_measurement(self):
        if not self.camera_ctrl.is_running:
            QMessageBox.warning(self, "Предупреждение", "Камера не подключена")
            return
        
        if self.dpi_recorder.is_recording:
            self.dpi_recorder.image_count = 0
            self.dpi_recorder.start_time = time.time()
        
        params = {
            'steps': int(self.steps_combo.currentText()),
            'lambda_angstrom': float(self.lambda_input.text()),
            'delay': self.delay_slider.value(),
            'unwrap': self.unwrap_checkbox.isChecked(),
            'remove_trend': self.trend_checkbox.isChecked(),
            'inverse': self.inverse_checkbox.isChecked(),
            'rainbow': self.rainbow_checkbox.isChecked(),
            'threshold': self.threshold_slider.value() / 100.0,
            'scale': True,
            'tile_unwrap': hasattr(self, 'tile_unwrap_checkbox') and self.tile_unwrap_checkbox.isChecked(),
            'tile_size': self.tile_size_slider.value(),
            'series_count': int(self.series_spin.value())
        }
        
        self.worker = MeasurementWorker(self.camera_ctrl, self.arduino_ctrl, params)
        self.worker.new_phase_image.connect(self.update_phase_image)
        self.worker.phase_data_ready.connect(self.on_phase_data_ready)
        self.worker.new_interferogram.connect(self.update_interferogram_image)
        # Отображение миниатюр шагов удалено
        self.worker.finished.connect(self.on_measurement_finished)
        self.worker.error.connect(self.on_measurement_error)
        self.worker.start()
        
        self.start_button.setText("Остановить измерение")

        # Очистка миниатюр не требуется

    def stop_measurement(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.start_button.setText("Начать измерение")

    @Slot(np.ndarray)
    def update_phase_image(self, cv_img):
        # Применяем полиномиальное удаление тренда если включено
        if hasattr(self, 'polynomial_trend_checkbox') and self.polynomial_trend_checkbox.isChecked():
            # Получаем фазовые данные из изображения (это упрощение)
            # В реальности нужно передавать фазовые данные отдельно
            pass
        
        # Сохраняем текущие данные для экспорта
        self.export_csv_button.setEnabled(True)
        
        # Сохранение DPI перенесено в on_phase_data_ready для надёжности
        
        # Отображаем изображение
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            q_image = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = cv_img.shape
            q_image = QImage(cv_img.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        self.last_phase_qimage = q_image
        if getattr(self, 'main_view_mode', 'camera') == 'phase':
            self.interferogram_view.set_image(q_image, preserve_transform=True)
        
        

    @Slot(np.ndarray)
    def update_interferogram_image(self, frame):
        if len(frame.shape) == 2:
            height, width = frame.shape
            q_image = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        else:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        if getattr(self, 'main_view_mode', 'camera') == 'camera':
            self.interferogram_view.set_image(q_image, preserve_transform=True)
        

    @Slot(np.ndarray)
    def on_phase_data_ready(self, phase_data):
        self.current_phase_data = phase_data
        if self.dpi_recorder.is_recording and phase_data is not None:
            self.dpi_recorder.save_phase_data(phase_data, None)

    def on_measurement_finished(self):
        self.start_button.setText("Начать измерение")
        if self.dpi_recorder.is_recording:
            self.dpi_recorder.mark_experiment_end()

    def on_measurement_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка измерения", error_msg)
        self.start_button.setText("Начать измерение")

    

    

    def save_settings(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить настройки", "settings.txt", "Текстовые файлы (*.txt)"
        )
        if filename:
            try:
                settings = [
                    str(self.camera_combo.currentIndex()),
                    str(self.steps_combo.currentIndex()),
                    str(self.lambda_input.text()),
                    str(self.inverse_checkbox.isChecked()),
                    str(self.rainbow_checkbox.isChecked()),
                    str(self.trend_checkbox.isChecked()),
                    str(self.delay_slider.value()),
                    f"{self.threshold_slider.value() / 100.0}",
                    str(self.unwrap_checkbox.isChecked()),
                    self.port_combo.currentText()
                ]
                with open(filename, 'w', encoding='utf-8') as f:
                    for s in settings:
                        f.write(s + "\n")
                QMessageBox.information(self, "Успех", f"Настройки сохранены: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")

    def load_settings(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить настройки", "", "Текстовые файлы (*.txt)"
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip() != ""]
                cam_idx = int(lines[0]) if len(lines) > 0 else 0
                steps_idx = int(lines[1]) if len(lines) > 1 else 1
                self.camera_combo.setCurrentIndex(min(cam_idx, self.camera_combo.count()-1))
                self.steps_combo.setCurrentIndex(min(steps_idx, self.steps_combo.count()-1))
                if len(lines) > 2:
                    self.lambda_input.setText(lines[2])
                if len(lines) > 3:
                    self.inverse_checkbox.setChecked(lines[3].lower() == 'true')
                if len(lines) > 4:
                    self.rainbow_checkbox.setChecked(lines[4].lower() == 'true')
                if len(lines) > 5:
                    self.trend_checkbox.setChecked(lines[5].lower() == 'true')
                if len(lines) > 6:
                    self.delay_slider.setValue(int(float(lines[6])))
                if len(lines) > 7:
                    val = float(lines[7])
                    self.threshold_slider.setValue(int(val * 100))
                    self.threshold_label.setText(f"{val:.2f}")
                if len(lines) > 8:
                    self.unwrap_checkbox.setChecked(lines[8].lower() == 'true')
                if len(lines) > 9:
                    port_text = lines[9]
                    idx = self.port_combo.findText(port_text)
                    if idx != -1:
                        self.port_combo.setCurrentIndex(idx)
                    else:
                        self.port_combo.addItem(port_text)
                        self.port_combo.setCurrentIndex(self.port_combo.count()-1)
                QMessageBox.information(self, "Успех", f"Настройки загружены: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить: {str(e)}")

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
                params = {
                    'steps': int(self.steps_combo.currentText()),
                    'lambda_angstrom': float(self.lambda_input.text()),
                    'threshold': self.threshold_slider.value() / 100.0,
                    'delay': self.delay_slider.value()
                }
                if self.dpi_recorder.start_recording(folder, params):
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
            ok, result = save_image_as_csv(self.current_phase_data, filename)
            if ok:
                QMessageBox.information(self, "Успех", f"Данные сохранены в {result}")
            else:
                QMessageBox.critical(self, "Ошибка", result)

    def closeEvent(self, event):
        """Обработчик закрытия окна."""
        # Останавливаем все активные процессы
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        if self.camera_stream_worker and self.camera_stream_worker.isRunning():
            self.camera_stream_worker.stop()
            
        if self.dpi_recorder.is_recording:
            self.dpi_recorder.stop_recording()
            
        # Отключаем контроллеры
        self.camera_ctrl.stop()
        self.arduino_ctrl.disconnect()
        
        event.accept()
    
    def toggle_camera_stream(self):
        """Переключает состояние трансляции камеры."""
        if self.camera_stream_worker and self.camera_stream_worker.isRunning():
            self.stop_camera_stream()
        else:
            self.start_camera_stream()
    
    def start_camera_stream(self):
        """Запускает трансляцию видео с камеры."""
        if not self.camera_ctrl.is_running:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите и подключите камеру")
            return
            
        try:
            self.camera_stream_worker = CameraStreamWorker(self.camera_ctrl)
            self.camera_stream_worker.new_frame.connect(self.update_camera_frame)
            self.camera_stream_worker.error.connect(self.on_camera_stream_error)
            self.camera_stream_worker.start()
            
            self.stream_button.setText("Остановить трансляцию")
            print("Трансляция камеры запущена")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить трансляцию: {e}")
    
    def stop_camera_stream(self):
        """Останавливает трансляцию видео с камеры."""
        if self.camera_stream_worker and self.camera_stream_worker.isRunning():
            self.camera_stream_worker.stop()
            self.stream_button.setText("Включить трансляцию")
            print("Трансляция камеры остановлена")
    
    @Slot(np.ndarray)
    def update_camera_frame(self, frame):
        """Обновляет отображение кадра с камеры."""
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            if getattr(self, 'main_view_mode', 'camera') == 'camera':
                self.interferogram_view.set_image(q_image, preserve_transform=True)
            
        except Exception as e:
            print(f"Ошибка обновления кадра: {e}")

    def toggle_main_view(self):
        if self.main_view_mode == 'camera':
            if self.last_phase_qimage is None:
                QMessageBox.warning(self, "Предупреждение", "Фазовое изображение ещё не получено")
                return
            self.main_view_mode = 'phase'
            self.toggle_view_button.setText("Показать камеру")
            self.interferogram_view.set_image(self.last_phase_qimage, preserve_transform=False)
        else:
            self.main_view_mode = 'camera'
            self.toggle_view_button.setText("Показать фазу")
    
    @Slot(str)
    def on_camera_stream_error(self, error_msg):
        """Обработчик ошибок трансляции камеры."""
        QMessageBox.critical(self, "Ошибка трансляции", error_msg)
        self.stop_camera_stream()

class GraphicsImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self._zoom = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def set_image(self, qimage: QImage, preserve_transform: bool = False):
        pixmap = QPixmap.fromImage(qimage)
        is_first = self._pix_item.pixmap().isNull()
        self._pix_item.setPixmap(pixmap)
        if preserve_transform and self._zoom != 0:
            return
        self.fitInView(self._pix_item, Qt.KeepAspectRatio)
        if not preserve_transform:
            self._zoom = 0

    def wheelEvent(self, event):
        if self._pix_item.pixmap().isNull():
            return
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)
        self._zoom += 1 if angle > 0 else -1
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pix_item.pixmap().isNull() and self._zoom == 0:
            self.fitInView(self._pix_item, Qt.KeepAspectRatio)

class ClickableThumb(QWidget):
    clicked = Signal(QImage)
    def __init__(self, qimg: QImage, caption: str):
        super().__init__()
        self.qimage = qimg
        self.label_img = QLabel()
        pix = QPixmap.fromImage(qimg).scaledToHeight(90, Qt.SmoothTransformation)
        self.label_img.setPixmap(pix)
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_txt = QLabel(caption)
        self.label_txt.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.label_img)
        lay.addWidget(self.label_txt)
    def mousePressEvent(self, e):
        self.clicked.emit(self.qimage)
