"""Identificación ligera del hardware usado en los experimentos."""

import os
import platform
import sys
from pathlib import Path


def cpu_model():
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def hardware_info(device, torch_module):
    return {
        "cpu_model": cpu_model(),
        "cpu_logical_count": os.cpu_count(),
        "platform": platform.platform(),
        "cuda_device": torch_module.cuda.get_device_name() if device == "cuda" else None,
    }
