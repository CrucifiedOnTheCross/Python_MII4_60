import importlib.util
import os
import sys

def _import_test_module():
    here = os.path.dirname(__file__)
    project_root = os.path.dirname(here)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    path = os.path.join(here, "test_arduino_blink.py")
    spec = importlib.util.spec_from_file_location("test_arduino_blink", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if __name__ == "__main__":
    mod = _import_test_module()
    mod.test_blink_serial_sequence_specific_pin()
    mod.test_blink_serial_sequence_default_pin()
    print("OK: blink serial sequence tests passed")
