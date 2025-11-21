# MII4_60_Python/config.py

# Ключевые параметры алгоритма
DEFAULT_STEPS = 3
DEFAULT_LAMBDA = 750  # Длина волны в нм (Java использует 6328, но в нм это 632.8)
DEFAULT_THRESHOLD = 0.8
DEFAULT_DELAY = 300     # Задержка в мс

# Настройки камеры
DEFAULT_CAMERA_RESOLUTION = (640, 480)
DEFAULT_CAMERA_FPS = 30

# Настройки Arduino
DEFAULT_ARDUINO_BAUDRATE = 57600
ARDUINO_PINS = list(range(2, 11))  # Пины 2-10 для управления

# Настройки обработки
DEFAULT_DELIMITER = 10  # Разделитель для специального unwrap
MAX_TILES_SIZE = (1000, 1000)  # Максимальный размер для tiles массива

# Настройки DPI записи
DPI_IMAGE_FORMAT = "PNG"
DPI_CSV_FORMAT = "CSV"

# Настройки интерферограмм
INTERFEROGRAM_METHODS = ["average", "first", "last"]
DEFAULT_INTERFEROGRAM_METHOD = "average"

# Пути к файлам
COLORMAP_FILE = "data/colorArray.csv"
SETTINGS_FILE = "data/settings.txt"
VALUES_FILE = "data/values.txt"

# Цветовые схемы
COLORMAP_SIZE = 1280  # Размер цветовой палитры

# Настройки GUI
GUI_WINDOW_SIZE = (800, 600)
CONTROLS_PANEL_WIDTH = 300