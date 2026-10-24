"""CPU Open Image Denoise bridge for exploratory Lavapipe rollouts.

The input is SAPIEN's display RGB, so this uses OIDN's LDR/sRGB color-only
filter. RoboTwin's native denoiser receives renderer buffers before display
conversion; this bridge does not claim pixel equivalence to that path.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np


class CpuOIDN:
    def __init__(self, library_path: str):
        if not Path(library_path).is_file():
            raise FileNotFoundError(library_path)
        lib = ct.CDLL(library_path)
        lib.oidnNewDevice.argtypes = [ct.c_int]
        lib.oidnNewDevice.restype = ct.c_void_p
        lib.oidnCommitDevice.argtypes = [ct.c_void_p]
        lib.oidnNewFilter.argtypes = [ct.c_void_p, ct.c_char_p]
        lib.oidnNewFilter.restype = ct.c_void_p
        lib.oidnSetSharedFilterImage.argtypes = [
            ct.c_void_p,
            ct.c_char_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
        ]
        lib.oidnSetFilterBool.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_bool]
        lib.oidnCommitFilter.argtypes = [ct.c_void_p]
        lib.oidnExecuteFilter.argtypes = [ct.c_void_p]
        lib.oidnGetDeviceError.argtypes = [ct.c_void_p, ct.POINTER(ct.c_char_p)]
        lib.oidnGetDeviceError.restype = ct.c_int
        lib.oidnReleaseFilter.argtypes = [ct.c_void_p]
        lib.oidnReleaseDevice.argtypes = [ct.c_void_p]
        self.lib = lib
        self.device = lib.oidnNewDevice(1)  # OIDN_DEVICE_TYPE_CPU
        if not self.device:
            raise RuntimeError("OIDN could not create its CPU device")
        lib.oidnCommitDevice(self.device)
        self._check_error()
        self._filters: dict[tuple[int, int], tuple[int, np.ndarray, np.ndarray]] = {}

    def _check_error(self) -> None:
        message = ct.c_char_p()
        code = self.lib.oidnGetDeviceError(self.device, ct.byref(message))
        if code:
            raise RuntimeError(f"OIDN error {code}: {message.value!r}")

    def _filter_for(self, width: int, height: int):
        key = width, height
        cached = self._filters.get(key)
        if cached is not None:
            return cached
        source = np.empty((height, width, 3), dtype=np.float32)
        output = np.empty_like(source)
        filt = self.lib.oidnNewFilter(self.device, b"RT")
        if not filt:
            self._check_error()
            raise RuntimeError("OIDN could not create RT filter")
        for name, image in ((b"color", source), (b"output", output)):
            self.lib.oidnSetSharedFilterImage(
                filt, name, image.ctypes.data, 3, width, height, 0, 0, 0
            )  # OIDN_FORMAT_FLOAT3
        self.lib.oidnSetFilterBool(filt, b"srgb", True)
        self.lib.oidnCommitFilter(filt)
        self._check_error()
        result = filt, source, output
        self._filters[key] = result
        return result

    def denoise(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"OIDN expected HxWx3 uint8 RGB, got {rgb.shape} {rgb.dtype}")
        height, width, _ = rgb.shape
        filt, source, output = self._filter_for(width, height)
        np.multiply(rgb, 1.0 / 255.0, out=source, casting="unsafe")
        self.lib.oidnExecuteFilter(filt)
        self._check_error()
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)

    def close(self) -> None:
        for filt, _, _ in self._filters.values():
            self.lib.oidnReleaseFilter(filt)
        self._filters.clear()
        if self.device:
            self.lib.oidnReleaseDevice(self.device)
            self.device = None

    def __del__(self) -> None:
        if getattr(self, "device", None):
            self.close()
