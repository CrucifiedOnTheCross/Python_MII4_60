import cv2
import serial
import serial.tools.list_ports
import time

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

    def start(self, index, resolution):
        """Запускает захват с выбранной камеры."""
        if self.is_running:
            self.stop()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.is_running = True
        print(f"Камера {index} запущена с разрешением {resolution}")

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

class ArduinoController:
    """Управляет соединением и отправкой команд на Arduino."""
    def __init__(self):
        self.ser = None
        self.is_connected = False

    @staticmethod
    def list_ports():
        """Возвращает список доступных COM-портов."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["Нет портов"]

    def connect(self, port, baudrate=57600):
        """Подключается к Arduino."""
        if self.is_connected:
            self.disconnect()
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.is_connected = True
            time.sleep(2) # Даем время на инициализацию соединения
            print(f"Подключено к {port}")
        except serial.SerialException as e:
            raise IOError(f"Не удалось подключиться к {port}: {e}")

    def disconnect(self):
        """Отключается от Arduino."""
        if self.ser:
            self.ser.close()
        self.ser = None
        self.is_connected = False
        
    def send_step_command(self, step_index):
        """Отправляет команду для выполнения шага сдвига фазы."""
        if not self.is_connected:
            print("Arduino не подключен.")
            return
        
        # Предполагаем, что Arduino ожидает байт с номером шага (0-4)
        # Этот код нужно адаптировать под прошивку Arduino
        # В оригинале использовались цифровые пины 2-10
        # Здесь для примера отправляем номер шага
        try:
            command = f"STEP:{step_index}\n".encode('utf-8')
            self.ser.write(command)
            print(f"Отправлена команда для шага {step_index}")
        except serial.SerialException as e:
            print(f"Ошибка отправки данных: {e}")