"""Pure Step Engines for science frame preparation.

These functions flat-field raw science exposures, propagate formal variance, and
combine nodded dither pairs (Pair Differencing, ``A - B``) or a single fixed
exposure (Stare Mode). They operate only on plain NumPy arrays (ADR 0003) and
are used by the ``process_science`` Step Engine.
"""

from __future__ import annotations

import numpy as np


def flat_field(image: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Divide an exposure by the normalized flat-field response.

    Pixels where the flat response is non-positive or non-finite are set to
    zero rather than producing infinities.

    Args:
        image: Raw exposure counts.
        flat: Normalized flat-field response, same shape as ``image``.

    Returns:
        The flat-fielded exposure counts.
    """
    image = np.asarray(image, dtype=float)
    flat = np.asarray(flat, dtype=float)
    good = np.isfinite(flat) & (flat > 0) & np.isfinite(image)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(image, flat, out=np.zeros_like(image), where=good)
    return out


def frame_variance(
    counts: np.ndarray,
    flat: np.ndarray,
    *,
    gain: float = 1.0,
    read_noise: float = 0.0,
) -> np.ndarray:
    """Propagate detector noise into the flat-fielded frame variance.

    The Poisson variance of the raw counts (plus read noise) is pushed through
    the division by the flat response.

    Args:
        counts: Raw exposure counts.
        flat: Normalized flat-field response, same shape as ``counts``.
        gain: Detector gain, in electrons per count.
        read_noise: Detector read noise, in electrons.

    Returns:
        The per-pixel variance of the flat-fielded frame.
    """
    counts = np.asarray(counts, dtype=float)
    flat = np.asarray(flat, dtype=float)
    gain = max(float(gain), 1e-12)
    var_counts = np.clip(counts, 0.0, None) / gain + float(read_noise) ** 2
    safe = np.where(np.isfinite(flat) & (flat > 0), flat, 1.0)
    return var_counts / safe**2


def pair_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Difference two nodded exposures to cancel the sky background."""
    return np.asarray(frame_a, dtype=float) - np.asarray(frame_b, dtype=float)


def combine_frames(frames: np.ndarray) -> np.ndarray:
    """Average a stack of science frames along the exposure axis.

    Used for both the mean of pair differences and the mean of flat-fielded
    stare frames; a 2D input is returned unchanged.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim == 2:
        return frames.copy()
    return np.nanmean(frames, axis=0)


__all__ = [
    "combine_frames",
    "flat_field",
    "frame_variance",
    "pair_difference",
]
