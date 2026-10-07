"""Small array utilities shared by the calibration Step Engines."""

from __future__ import annotations

import numpy as np


def boxcar_1d(values: np.ndarray, size: int) -> np.ndarray:
    """Smooth a 1D array with a normalized boxcar of ``size`` pixels."""
    values = np.asarray(values, dtype=float)
    if size <= 1:
        return values.copy()
    kernel = np.ones(size, dtype=float) / size
    return np.convolve(values, kernel, mode="same")
