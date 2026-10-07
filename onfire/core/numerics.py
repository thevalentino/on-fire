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


def gaussian_filter1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1D array with a normalized Gaussian of width ``sigma`` pixels.

    The kernel is truncated at four sigma and edge samples are handled by
    ``mode='same'`` convolution, which is sufficient for the object-detection
    profiles where a small bias near the array ends is unimportant.
    """
    values = np.asarray(values, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


__all__ = ["boxcar_1d", "gaussian_filter1d"]
