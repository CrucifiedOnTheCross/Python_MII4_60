import time
import config
from hardware.controller import ArduinoController


class DummySerial:
    def __init__(self):
        self.writes = []

    def write(self, data: bytes):
        self.writes.append(data)


def test_blink_serial_sequence_specific_pin():
    ctrl = ArduinoController()
    ctrl.is_connected = True
    ctrl._mode = 'serial'
    ctrl.ser = DummySerial()

    ok = ctrl.blink(pin=7, duration_ms=1)
    assert ok is True
    seq = [b.decode('utf-8') for b in ctrl.ser.writes]
    assert seq == [
        "PINMODE:7:OUTPUT\n",
        "DIGITAL:7:HIGH\n",
        "DIGITAL:7:LOW\n",
    ]


def test_blink_serial_sequence_default_pin():
    ctrl = ArduinoController()
    ctrl.is_connected = True
    ctrl._mode = 'serial'
    ctrl.ser = DummySerial()

    ok = ctrl.blink(duration_ms=1)
    assert ok is True
    seq = [b.decode('utf-8') for b in ctrl.ser.writes]
    assert seq == [
        f"PINMODE:{config.BLINK_PIN}:OUTPUT\n",
        f"DIGITAL:{config.BLINK_PIN}:HIGH\n",
        f"DIGITAL:{config.BLINK_PIN}:LOW\n",
    ]

