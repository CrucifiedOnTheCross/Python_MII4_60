import cv2
import serial
import serial.tools.list_ports
import time
import config
try:
    import pyfirmata
    from pyfirmata import util
    _HAS_FIRMATA = True
except Exception:
    _HAS_FIRMATA = False

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
        self.board = None
        self.is_connected = False
        self.current_step = 0
        self._mode = None  # 'firmata' or 'serial'

    @staticmethod
    def list_ports():
        """Возвращает список доступных COM-портов."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            return ["Нет доступных портов"]
        items = []
        for p in ports:
            desc = p.description if hasattr(p, 'description') else ''
            items.append(f"{p.device} {desc}".strip())
        return items

    def connect(self, port, baudrate=None):
        """Подключается к Arduino. Предпочтительно через Firmata, как в Java-версии."""
        if baudrate is None:
            baudrate = config.DEFAULT_ARDUINO_BAUDRATE

        if self.is_connected:
            self.disconnect()
        
        # Пытаемся использовать Firmata, если библиотека доступна
        if _HAS_FIRMATA:
            try:
                self.board = pyfirmata.Arduino(port)
                # Запускаем итератор, чтобы не переполнялся буфер
                util.Iterator(self.board).start()
                self._mode = 'firmata'
                self.is_connected = True
                time.sleep(2)
                self._initialize_pins()
                print(f"Подключено по Firmata к {port}")
                return True
            except Exception as e:
                print(f"Не удалось подключиться по Firmata: {e}. Пытаемся через serial...")
        
        # Фолбэк: простой serial с текстовым протоколом
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self._mode = 'serial'
            self.is_connected = True
            time.sleep(2)
            self._initialize_pins()
            print(f"Подключено по serial к {port} на скорости {baudrate}")
            return True
        except serial.SerialException as e:
            raise IOError(f"Не удалось подключиться к {port}: {e}")

    def disconnect(self):
        """Отключается от Arduino."""
        try:
            if self._mode == 'firmata' and self.board:
                self.board.exit()
            if self.ser:
                self.ser.close()
        finally:
            self.board = None
            self.ser = None
        self.is_connected = False
        
    def _initialize_pins(self):
        """Инициализирует пины Arduino как OUTPUT и устанавливает начальное состояние."""
        if not self.is_connected:
            return
            
        try:
            if self._mode == 'firmata' and self.board:
                for pin in config.ARDUINO_PINS:
                    self.board.digital[pin].mode = pyfirmata.OUTPUT
                    self.board.digital[pin].write(0)
                    time.sleep(0.005)
                self.board.digital[8].write(1)
            elif self._mode == 'serial' and self.ser:
                for pin in config.ARDUINO_PINS:
                    command = f"PINMODE:{pin}:OUTPUT\n".encode('utf-8')
                    self.ser.write(command)
                    time.sleep(0.005)
                    command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                    self.ser.write(command)
                    time.sleep(0.005)
                command = f"DIGITAL:8:HIGH\n".encode('utf-8')
                self.ser.write(command)
            else:
                print("Неизвестный режим соединения для инициализации пинов")
        except Exception as e:
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
            # Сначала все LOW
            target_pin = step_index + 2
            if self._mode == 'firmata' and self.board:
                for pin in range(2, 11):
                    self.board.digital[pin].write(0)
                if 2 <= target_pin <= 10:
                    self.board.digital[target_pin].write(1)
                    self.current_step = step_index
                    print(f"Выполнен шаг {step_index}, пин {target_pin} HIGH (Firmata)")
                else:
                    print(f"Неверный индекс шага: {step_index}")
            elif self._mode == 'serial' and self.ser:
                for pin in range(2, 11):
                    command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                    self.ser.write(command)
                    time.sleep(0.001)
                if 2 <= target_pin <= 10:
                    command = f"DIGITAL:{target_pin}:HIGH\n".encode('utf-8')
                    self.ser.write(command)
                    self.current_step = step_index
                    print(f"Выполнен шаг {step_index}, пин {target_pin} HIGH (serial)")
                else:
                    print(f"Неверный индекс шага: {step_index}")
            else:
                print("Неизвестный режим соединения для команды шага")
                
        except Exception as e:
            print(f"Ошибка отправки команды шага: {e}")
    
    def reset_all_pins(self):
        """Сбрасывает все пины в LOW состояние."""
        if not self.is_connected:
            return
            
        try:
            if self._mode == 'firmata' and self.board:
                for pin in config.ARDUINO_PINS:
                    self.board.digital[pin].write(0)
            elif self._mode == 'serial' and self.ser:
                for pin in config.ARDUINO_PINS:
                    command = f"DIGITAL:{pin}:LOW\n".encode('utf-8')
                    self.ser.write(command)
                    time.sleep(0.001)
            else:
                print("Неизвестный режим соединения для сброса пинов")
        except Exception as e:
            print(f"Ошибка сброса пинов: {e}")
    
    def get_current_step(self):
        """Возвращает текущий активный шаг."""
        return self.current_step