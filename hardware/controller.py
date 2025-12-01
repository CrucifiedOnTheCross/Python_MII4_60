import cv2
import serial
import serial.tools.list_ports
import time
import config
import inspect
from collections import namedtuple

if not hasattr(inspect, 'getargspec'):
    _ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
    def _compat_getargspec(func):
        fs = inspect.getfullargspec(func)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    inspect.getargspec = _compat_getargspec
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
        ports = serial.tools.list_ports.comports()
        if not ports:
            return ["Нет доступных портов"]
        result = []
        for p in ports:
            dev = p.device
            desc = p.description if hasattr(p, 'description') else ''
            busy = False
            try:
                t = serial.Serial(dev, 9600, timeout=0.2)
                t.close()
            except Exception:
                busy = True
            suffix = " (Занят)" if busy else ""
            result.append(f"{dev} {desc}{suffix}".strip())
        return result

    def connect_auto(self, baudrate=None):
        if baudrate is None:
            baudrate = config.DEFAULT_ARDUINO_BAUDRATE
        ports = serial.tools.list_ports.comports()
        preferred = []
        fallback = []
        for p in ports:
            dev = p.device
            desc = p.description if hasattr(p, 'description') else ''
            try:
                t = serial.Serial(dev, 9600, timeout=0.2)
                t.close()
                if any(x in desc for x in ["Arduino", "USB", "CH340", "CDC"]):
                    preferred.append(dev)
                else:
                    fallback.append(dev)
            except Exception:
                continue
        for dev in preferred + fallback:
            try:
                if self.connect(dev, baudrate):
                    return True
            except Exception:
                continue
        return False

    def connect(self, port, baudrate=None):
        """Подключается к Arduino. Предпочтительно через Firmata, как в Java-версии."""
        if baudrate is None:
            baudrate = config.DEFAULT_ARDUINO_BAUDRATE

        if self.is_connected:
            self.disconnect()
        
        # Пытаемся использовать Firmata, если библиотека доступна
        if _HAS_FIRMATA and not self._is_esp_board(port):
            tmp_board = None
            try:
                tmp_board = pyfirmata.Arduino(port)
                util.Iterator(tmp_board).start()
                time.sleep(2)
                # Присваиваем только после успешной инициализации
                self.board = tmp_board
                self._mode = 'firmata'
                self.is_connected = True
                self._initialize_pins()
                print(f"Подключено по Firmata к {port}")
                return True
            except PermissionError as e:
                print(f"Доступ к порту отклонён: {e}")
                time.sleep(1.5)
                try:
                    if tmp_board:
                        tmp_board.exit()
                except Exception:
                    pass
            except Exception as e:
                # Гарантируем освобождение ресурса при частичной инициализации
                try:
                    if tmp_board:
                        tmp_board.exit()
                except Exception:
                    pass
                print(f"Не удалось подключиться по Firmata: {e}. Пытаемся через serial...")
        
        # Фолбэк: простой serial с текстовым протоколом
        last_err = None
        is_esp = self._is_esp_board(port)
        for br in [baudrate, 115200, 9600]:
            tmp_ser = None
            try:
                tmp_ser = serial.Serial(port, br, timeout=1, rtscts=False, dsrdtr=(False if is_esp else True), write_timeout=1)
                time.sleep(2)
                # Присваиваем только после успешной инициализации
                self.ser = tmp_ser
                self._mode = 'serial'
                self.is_connected = True
                self._initialize_pins()
                print(f"Подключено по serial к {port} на скорости {br}")
                return True
            except PermissionError as e:
                last_err = e
                print(f"Доступ к порту отклонён: {e}")
                time.sleep(1.5)
                try:
                    if tmp_ser:
                        tmp_ser.close()
                except Exception:
                    pass
            except serial.SerialException as e:
                last_err = e
                try:
                    if tmp_ser:
                        tmp_ser.close()
                except Exception:
                    pass
            except Exception as e:
                last_err = e
                # Закрываем дескриптор при любых исключениях в процессе инициализации
                try:
                    if tmp_ser:
                        tmp_ser.close()
                except Exception:
                    pass
        raise IOError(f"Не удалось подключиться к {port}: {last_err}")

    @staticmethod
    def _get_port_info(port):
        for p in serial.tools.list_ports.comports():
            if p.device == port:
                return p
        return None

    def _is_esp_board(self, port):
        info = self._get_port_info(port)
        if not info:
            return False
        desc = getattr(info, 'description', '') or ''
        manu = getattr(info, 'manufacturer', '') or ''
        hwid = getattr(info, 'hwid', '') or ''
        text = f"{desc} {manu} {hwid}".upper()
        return ('ESP' in text) or ('NODEMCU' in text) or ('WEMOS' in text)

    def disconnect(self):
        """Отключается от Arduino."""
        try:
            if self._mode == 'firmata' and self.board:
                self.board.exit()
            if self.ser:
                self.ser.close()
                # Небольшая пауза для корректного освобождения порта на Windows
                time.sleep(0.2)
        finally:
            self.board = None
            self.ser = None
            self._mode = None
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

    def blink(self, pin=None, duration_ms=300):
        """Коротко включает пин и выключает для проверки связи.
        По умолчанию мигает встроенным светодиодом на `config.BLINK_PIN`.
        """
        if not self.is_connected:
            print("Arduino не подключен.")
            return False

        if pin is None:
            pin = getattr(config, 'BLINK_PIN', 13)
            # Для ESP используем другой пин по умолчанию
            try:
                if self._mode == 'serial' and self.ser:
                    port = getattr(self.ser, 'port', None)
                    if port and self._is_esp_board(port):
                        pin = getattr(config, 'ESP_BLINK_PIN', 2)
            except Exception:
                pass

        try:
            if self._mode == 'firmata' and self.board:
                self.board.digital[pin].mode = pyfirmata.OUTPUT
                self.board.digital[pin].write(1)
                time.sleep(duration_ms / 1000.0)
                self.board.digital[pin].write(0)
                return True
            elif self._mode == 'serial' and self.ser:
                invert = False
                try:
                    port = getattr(self.ser, 'port', None)
                    if port and self._is_esp_board(port):
                        invert = getattr(config, 'ESP_LED_INVERTED', True)
                except Exception:
                    pass
                self.ser.write(f"PINMODE:{pin}:OUTPUT\n".encode('utf-8'))
                time.sleep(0.005)
                if invert:
                    self.ser.write(f"DIGITAL:{pin}:LOW\n".encode('utf-8'))
                    time.sleep(duration_ms / 1000.0)
                    self.ser.write(f"DIGITAL:{pin}:HIGH\n".encode('utf-8'))
                else:
                    self.ser.write(f"DIGITAL:{pin}:HIGH\n".encode('utf-8'))
                    time.sleep(duration_ms / 1000.0)
                    self.ser.write(f"DIGITAL:{pin}:LOW\n".encode('utf-8'))
                return True
            else:
                print("Неизвестный режим соединения для мигания")
                return False
        except Exception as e:
            print(f"Ошибка мигания пина {pin}: {e}")
            return False
