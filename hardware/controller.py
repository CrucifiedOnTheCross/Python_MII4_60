import cv2
import serial
import serial.tools.list_ports
import time
import config

class CameraController:
    """Управляет захватом видео с камеры."""
    def __init__(self):
        self.cap = None
        self.is_running = False

    @staticmethod
    def list_cameras():
        """Возвращает список доступных камер."""
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.read()[0]:
                break
            arr.append(f"Камера {index}")
            cap.release()
            index += 1
        return arr if arr else ["Нет доступных камер"]

    def start(self, index, resolution=None, fps=None):
        """Запускает захват с выбранной камеры."""
        if self.is_running:
            self.stop()
            
        if resolution is None:
            resolution = config.DEFAULT_CAMERA_RESOLUTION
        if fps is None:
            fps = config.DEFAULT_CAMERA_FPS
            
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.is_running = True
        print(f"Камера {index} запущена с разрешением {resolution} и FPS {fps}")

    def stop(self):
        """Останавливает захват."""
        if self.cap:
            self.cap.release()
        self.cap = None
        self.is_running = False

    def get_frame(self):
        """Получает один кадр с камеры."""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Конвертируем в оттенки серого, как в оригинальном проекте
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return None
    
    def get_resolution(self):
        """Возвращает текущее разрешение камеры."""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return None

class ArduinoController:
    """Управляет соединением и отправкой команд на Arduino."""
    def __init__(self):
        self.ser = None
        self.is_connected = False
        self.current_step = 0

    @staticmethod
    def list_ports():
        """Возвращает список доступных COM-портов."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["Нет портов"]

    def connect(self, port, baudrate=None):
        """Подключается к Arduino."""
        if baudrate is None:
            baudrate = config.DEFAULT_ARDUINO_BAUDRATE
            
        if self.is_connected:
            self.disconnect()
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.is_connected = True
            time.sleep(2)  # Даем время на инициализацию соединения
            
            # Инициализируем пины как в Java версии
            self._initialize_pins()
            print(f"Подключено к {port} на скорости {baudrate}")
        except serial.SerialException as e:
            raise IOError(f"Не удалось подключиться к {port}: {e}")

    def disconnect(self):
        """Отключается от Arduino."""
        if self.ser:
            self.ser.close()
        self.ser = None
        self.is_connected = False
        
    def _initialize_pins(self):
        """Инициализирует пины Arduino как OUTPUT и устанавливает начальное состояние."""
        if not self.is_connected:
            return
            
        try:
            # Устанавливаем все пины в LOW состояние
            for pin in config.ARDUINO_PINS:
                command = f"PINMODE:{pin}:OUTPUT\n".encode('utf-8')
                self.ser.write(command)
                time.sleep(0.01)
                
                command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                self.ser.write(command)
                time.sleep(0.01)
            
            # Устанавливаем пин 8 в HIGH как в оригинале
            command = f"DIGITAL:8:HIGH\n".encode('utf-8')
            self.ser.write(command)
            
        except serial.SerialException as e:
            print(f"Ошибка инициализации пинов: {e}")
        
    def send_step_command(self, step_index):
        """
        Отправляет команду для выполнения шага сдвига фазы.
        Реализует логику функции step() из Java версии.
        """
        if not self.is_connected:
            print("Arduino не подключен.")
            return
        
        try:
            # Сначала устанавливаем все пины 2-10 в LOW (как в оригинале)
            for pin in range(2, 11):
                command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                self.ser.write(command)
                time.sleep(0.001)
            
            # Затем устанавливаем нужный пин в HIGH
            target_pin = step_index + 2  # step_index 0-4 соответствует пинам 2-6
            if 2 <= target_pin <= 10:
                command = f"DIGITAL:{target_pin}:HIGH\n".encode('utf-8')
                self.ser.write(command)
                self.current_step = step_index
                print(f"Выполнен шаг {step_index}, пин {target_pin} установлен в HIGH")
            else:
                print(f"Неверный индекс шага: {step_index}")
                
        except serial.SerialException as e:
            print(f"Ошибка отправки команды шага: {e}")
    
    def reset_all_pins(self):
        """Сбрасывает все пины в LOW состояние."""
        if not self.is_connected:
            return
            
        try:
            for pin in config.ARDUINO_PINS:
                command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                self.ser.write(command)
                time.sleep(0.001)
        except serial.SerialException as e:
            print(f"Ошибка сброса пинов: {e}")
    
    def get_current_step(self):
        """Возвращает текущий активный шаг."""
        return self.current_step