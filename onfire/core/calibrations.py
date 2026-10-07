"""Pure Step Engines for calibration frame preparation.

These functions operate only on plain NumPy arrays, as required by ADR 0003.
They combine raw flat-field exposures into a master flat and construct a
detector bad-pixel mask.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import mad_std

from onfire.core.numerics import boxcar_1d

_DEFAULT_SIGREJ: dict[int, float] = {3: 1.1, 4: 1.3, 5: 1.6, 6: 1.9}


def _default_sigrej(nimages: int) -> float:
    return _DEFAULT_SIGREJ.get(nimages, 2.0)


def _boxcar_rows(image: np.ndarray, size: int) -> np.ndarray:
    """Smooth each row of a 2D array with a boxcar of ``size`` pixels."""
    if size <= 1:
        return image.astype(float, copy=True)
    return np.vstack([boxcar_1d(row, size) for row in image])


def _boxcar_columns(image: np.ndarray, size: int) -> np.ndarray:
    """Smooth each column of a 2D array with a boxcar of ``size`` pixels."""
    if size <= 1:
        return image.astype(float, copy=True)
    return np.column_stack([boxcar_1d(column, size) for column in image.T])


def combine_flats(
    flats: np.ndarray,
    *,
    sigma_clip: bool = True,
    sigrej: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine a stack of raw flat-field exposures.

    Args:
        flats: Image stack with shape ``(nimages, nspec, nspat)``.
        sigma_clip: Reject pixels deviating from the per-pixel median when at
            least three images are combined. Sigma clipping needs at least
            three images to estimate a robust scale.
        sigrej: Rejection threshold in robust sigma. Defaults to a value based
            on the number of images.

    Returns:
        ``(combined, variance, good, nused)`` where ``combined`` and
        ``variance`` are the mean image and its propagated Poisson variance,
        ``good`` is the boolean good-pixel mask of the combination, and
        ``nused`` counts the exposures contributing to each pixel.
    """
    stack = np.asarray(flats, dtype=float)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(
            f"flats must have shape (nimages, nspec, nspat), got {stack.shape}"
        )

    nimages = stack.shape[0]
    good = np.isfinite(stack)

    if sigma_clip and nimages >= 3:
        if sigrej is None:
            sigrej = _default_sigrej(nimages)
        masked = np.where(good, stack, np.nan)
        median = np.nanmedian(masked, axis=0)
        dispersion = 1.4826 * np.nanmedian(np.abs(masked - median), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = np.abs(stack - median) / np.where(dispersion > 0, dispersion, np.inf)
        good &= (deviation < sigrej) | (dispersion == 0)

    effective = good.astype(float)
    weight_sum = effective.sum(axis=0)
    safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    values = np.where(good, stack, 0.0)

    combined = (values * effective).sum(axis=0) / safe_sum
    variance = (np.abs(stack) * effective**2).sum(axis=0) / safe_sum**2

    good_all = good.any(axis=0)
    nused = good.sum(axis=0)
    return combined, variance, good_all, nused


def build_bad_pixel_mask(
    flat: np.ndarray,
    *,
    n_sigma: float = 5.0,
    window: int = 5,
    flag_nonpositive: bool = True,
) -> np.ndarray:
    """Construct a detector bad-pixel mask from a combined flat.

    Args:
        flat: Combined flat image with shape ``(nspec, nspat)``.
        n_sigma: Rejection threshold, in robust sigma, for pixels deviating
            from the locally smoothed flat.
        window: Boxcar size, in pixels along the spatial axis, for the local
            smoothing used to find outliers.
        flag_nonpositive: Also flag non-finite and non-positive pixels, which
            have no usable flat-field response.

    Returns:
        A boolean mask with the same shape as ``flat``; ``True`` marks a bad
        pixel.
    """
    flat = np.asarray(flat, dtype=float)
    bad = ~np.isfinite(flat)
    if flag_nonpositive:
        bad |= flat <= 0

    smoothed = _boxcar_rows(np.where(np.isfinite(flat), flat, 0.0), window)
    residual = flat - smoothed
    valid = ~bad & np.isfinite(residual)
    if valid.any():
        sigma = mad_std(residual[valid])
        if sigma > 0:
            bad |= np.abs(residual) > n_sigma * sigma
    return bad


def normalize_flat(flat: np.ndarray, *, window: int = 101) -> np.ndarray:
    """Normalize a flat by its smooth spectral response.

    The response is estimated by smoothing along the dispersion axis (axis 0)
    so the spatial order structure is preserved for edge tracing.
    """
    flat = np.asarray(flat, dtype=float)
    window = max(1, min(window, flat.shape[0]))
    smooth = _boxcar_columns(flat, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.divide(flat, smooth, out=np.ones_like(flat), where=smooth != 0)
    return norm
