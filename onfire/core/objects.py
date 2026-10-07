"""Pure Step Engines for spatial object detection and tracing.

The functions here rectify each echelle order along its edge traces, collapse
the rectified image into a sigma-clipped spatial profile, detect object peaks
in that profile, and trace the object centroids along the dispersion axis. They
operate only on plain NumPy arrays (ADR 0003) and are used by the
``find_objects`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onfire.core.numerics import gaussian_filter1d


@dataclass
class ObjectTrace:
    """A traced object within a single echelle order.

    Attributes:
        order: Echelle order number the object lives in.
        coeff: Legendre coefficients of the spatial trace vs. dispersion.
        xmid: Dispersion-coordinate midpoint used for normalization.
        xscale: Dispersion-coordinate scale used for normalization.
        rms: RMS of the trace fit, in pixels.
        fracpos: Fractional position of the object at the order centre.
        spat_pixpos: Spatial pixel position of the object at the order centre.
        fwhm: Estimated spatial FWHM, in pixels.
        box_radius: Boxcar extraction/background radius, in pixels.
        snr: S/N of the detected profile peak.
        masked: Whether the trace fit failed.
    """

    order: int
    coeff: np.ndarray
    xmid: float
    xscale: float
    rms: float
    fracpos: float
    spat_pixpos: float
    fwhm: float
    box_radius: float
    snr: float
    masked: bool = False


def trace_coordinate(nspec: int) -> tuple[float, float]:
    """Return the Legendre dispersion-coordinate midpoint and scale."""
    xmid = (nspec - 1) / 2.0
    return xmid, (xmid if xmid > 0 else 1.0)


def rectify_order(
    image: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    nsamp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify an order onto a uniform fractional-spatial grid.

    Args:
        image: Detector image with shape ``(nspec, nspat)``.
        left: Left spatial edge, in pixels, per dispersion row.
        right: Right spatial edge, in pixels, per dispersion row.
        nsamp: Number of samples across the order.

    Returns:
        ``(rect, spat)`` where ``rect`` has shape ``(nspec, nsamp)`` and
        ``spat`` holds the detector spatial coordinate of each sample.
    """
    image = np.asarray(image, dtype=float)
    nspec, nspat = image.shape
    frac = (np.arange(nsamp, dtype=float) + 0.5) / nsamp
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    width = right - left
    spat = left[:, None] + width[:, None] * frac[None, :]
    detector = np.arange(nspat, dtype=float)
    rect = np.full((nspec, nsamp), np.nan)
    for row in range(nspec):
        valid = np.isfinite(spat[row])
        if not valid.any():
            continue
        rect[row, valid] = np.interp(
            spat[row, valid], detector, image[row], left=np.nan, right=np.nan
        )
    return rect, spat


def collapse_profile(
    rect: np.ndarray,
    *,
    gpm: np.ndarray | None = None,
    ndeg: int = 2,
    sigclip: float = 5.0,
    niter: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a rectified order into a continuum-subtracted spatial profile.

    Pixels are sigma-clipped along the dispersion axis, the surviving samples
    are averaged into a mean spatial profile, and a low-order polynomial is
    subtracted so the object peaks sit on a flat baseline.

    Args:
        rect: Rectified order image, shape ``(nspec, nsamp)``.
        gpm: Optional good-pixel mask, same shape as ``rect``.
        ndeg: Degree of the baseline polynomial to subtract.
        sigclip: Rejection threshold in robust sigma.
        niter: Maximum number of rejection iterations.

    Returns:
        ``(profile, good)`` with the baseline-subtracted profile and the final
        good-pixel mask.
    """
    rect = np.asarray(rect, dtype=float)
    data = np.where(np.isfinite(rect), rect, np.nan)
    good = np.isfinite(rect) if gpm is None else (np.asarray(gpm, dtype=bool) & np.isfinite(rect))

    for _ in range(max(1, niter)):
        masked = np.where(good, data, np.nan)
        median = np.nanmedian(masked, axis=0)
        dispersion = 1.4826 * np.nanmedian(np.abs(masked - median), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = np.abs(masked - median) / np.where(dispersion > 0, dispersion, np.inf)
        new_good = good & ((deviation <= sigclip) | (dispersion == 0))
        if np.array_equal(new_good, good):
            break
        good = new_good

    npix = good.sum(axis=0)
    flux = np.nansum(np.where(good, data, 0.0), axis=0) / np.where(npix > 0, npix, 1)
    flux = np.where(npix > 0, flux, np.nan)

    nsamp = rect.shape[1]
    x = np.arange(nsamp, dtype=float)
    finite = np.isfinite(flux)
    if finite.sum() > ndeg + 1:
        coeff = np.polyfit(x[finite], flux[finite], ndeg)
        baseline = np.polyval(coeff, x)
    else:
        baseline = np.nanmedian(flux) if finite.any() else 0.0
    profile = np.where(finite, flux - baseline, np.nan)
    return profile, good


def profile_snr(
    rect_ivar: np.ndarray,
    good: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    """Propagate inverse variance into a S/N spatial profile.

    Args:
        rect_ivar: Rectified inverse-variance image, shape ``(nspec, nsamp)``.
        good: The final good-pixel mask from :func:`collapse_profile`.
        profile: The baseline-subtracted spatial profile.

    Returns:
        The per-sample signal-to-noise of the collapsed profile.
    """
    rect_ivar = np.asarray(rect_ivar, dtype=float)
    good = np.asarray(good, dtype=bool)
    npix = good.sum(axis=0)
    valid_ivar = good & np.isfinite(rect_ivar) & (rect_ivar > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_pixels = np.divide(
            1.0, rect_ivar, out=np.zeros_like(rect_ivar), where=valid_ivar
        )
    var_sum = var_pixels.sum(axis=0)
    var = var_sum / np.where(npix > 0, npix, 1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = profile / np.sqrt(var)
    return np.where(np.isfinite(snr), snr, 0.0)


def find_profile_peaks(
    profile: np.ndarray,
    *,
    snr: np.ndarray | None = None,
    threshold: float = 5.0,
    fwhm: float = 3.0,
    min_sep_frac: float = 0.75,
    trim_edg: float = 1.0,
    max_objects: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect object peaks in a continuum-subtracted spatial profile.

    The profile (or its S/N) is smoothed with a Gaussian matched to ``fwhm``,
    local maxima above ``threshold`` are collected, peaks within ``trim_edg``
    samples of either edge are discarded, and the survivors are thinned to a
    minimum separation of ``min_sep_frac * fwhm``.

    Returns:
        ``(peaks, peak_snr)`` arrays of sample indices and their S/N values.
    """
    profile = np.asarray(profile, dtype=float)
    nsamp = profile.size
    sigma = max(float(fwhm) / 2.3548, 0.5)
    smoothed = gaussian_filter1d(np.nan_to_num(profile, nan=0.0), sigma)

    if snr is not None:
        smoothed_snr = gaussian_filter1d(np.nan_to_num(np.asarray(snr, float), nan=0.0), sigma)
    else:
        scatter = 1.4826 * np.median(np.abs(profile[np.isfinite(profile)] - np.median(profile[np.isfinite(profile)]))) if np.isfinite(profile).any() else 0.0
        if scatter <= 0:
            scatter = float(np.nanstd(profile)) or 1.0
        smoothed_snr = smoothed / scatter

    interior = (smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:])
    candidates = np.where(interior)[0] + 1
    candidates = candidates[smoothed[candidates] > 0]
    candidates = candidates[smoothed_snr[candidates] >= float(threshold)]
    candidates = candidates[
        (candidates >= trim_edg) & (candidates <= nsamp - 1 - trim_edg)
    ]

    if candidates.size == 0:
        return candidates, np.array([], dtype=float)

    min_sep = max(1.0, min_sep_frac * float(fwhm))
    order = candidates[np.argsort(-smoothed[candidates])]
    kept: list[int] = []
    for peak in order:
        if all(abs(peak - chosen) >= min_sep for chosen in kept):
            kept.append(int(peak))
    peaks = np.sort(np.asarray(kept, dtype=int))

    if max_objects is not None and peaks.size > max_objects:
        ranked = peaks[np.argsort(-smoothed[peaks])][:max_objects]
        peaks = np.sort(ranked)
    return peaks, smoothed_snr[peaks]


def _row_residual(
    rect: np.ndarray,
    valid: np.ndarray,
    *,
    ndeg: int = 1,
) -> np.ndarray:
    """Remove a per-row polynomial baseline, returning clipped residuals."""
    nspec, nsamp = rect.shape
    x = np.arange(nsamp, dtype=float)
    residual = np.zeros_like(rect)
    for row in range(nspec):
        good = valid[row] & np.isfinite(rect[row])
        if good.sum() <= ndeg + 1:
            continue
        coeff = np.polyfit(x[good], rect[row][good], ndeg)
        residual[row] = np.clip(rect[row] - np.polyval(coeff, x), 0.0, None)
    return residual


def trace_object(
    rect: np.ndarray,
    spat: np.ndarray,
    valid: np.ndarray,
    peak_index: int,
    *,
    half_width: float,
    maxshift: float = 1.0,
    ref_row: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """Trace one object's spatial centroid along the dispersion axis.

    Args:
        rect: Rectified order image, shape ``(nspec, nsamp)``.
        spat: Detector spatial coordinate of each rectified sample.
        valid: Good-pixel mask of the rectified image.
        peak_index: Sample index of the object peak at the reference row.
        half_width: Half-width of the centroid window, in pixels.
        maxshift: Maximum centroid shift between adjacent rows, in pixels.
        ref_row: Reference dispersion row; defaults to the middle row.

    Returns:
        ``(positions, xmid, xscale)`` with the per-row centroid positions and
        the dispersion-coordinate normalization.
    """
    rect = np.asarray(rect, dtype=float)
    spat = np.asarray(spat, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    nspec = rect.shape[0]
    if ref_row is None:
        ref_row = nspec // 2

    residual = _row_residual(rect, valid)
    positions = np.full(nspec, np.nan)

    def _step(row: int, expected: float) -> float:
        row_valid = valid[row] & np.isfinite(spat[row])
        lo = expected - half_width
        hi = expected + half_width
        window = row_valid & (spat[row] >= lo) & (spat[row] <= hi)
        if not window.any():
            return expected
        values = residual[row][window]
        coords = spat[row][window]
        total = values.sum()
        if total <= 0:
            return expected
        centroid = float((coords * values).sum() / total)
        return float(np.clip(centroid, expected - maxshift, expected + maxshift))

    expected = float(spat[ref_row, peak_index])
    if not np.isfinite(expected):
        expected = float(np.nanmedian(spat[ref_row]))
    positions[ref_row] = _step(ref_row, expected)
    previous = positions[ref_row]
    for row in range(ref_row + 1, nspec):
        expected = float(spat[row, peak_index]) if peak_index < spat.shape[1] else previous
        if not np.isfinite(expected):
            expected = previous
        previous = _step(row, expected)
        positions[row] = previous
    previous = positions[ref_row]
    for row in range(ref_row - 1, -1, -1):
        expected = float(spat[row, peak_index]) if peak_index < spat.shape[1] else previous
        if not np.isfinite(expected):
            expected = previous
        previous = _step(row, expected)
        positions[row] = previous

    xmid, xscale = trace_coordinate(nspec)
    return positions, xmid, xscale


def fit_trace(
    positions: np.ndarray,
    *,
    ncoeff: int,
    ref_row: int | None = None,
    sigrej: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Fit a 1D Legendre polynomial to object centroid positions.

    Returns:
        ``(coeff, rms)``; a failed fit yields all-NaN coefficients and NaN RMS.
    """
    positions = np.asarray(positions, dtype=float)
    nspec = positions.size
    if ref_row is None:
        ref_row = nspec // 2
    xmid, xscale = trace_coordinate(nspec)
    rows = np.arange(nspec, dtype=float)
    x = (rows - xmid) / xscale

    good = np.isfinite(positions)
    coeff = np.full(ncoeff, np.nan)
    for _ in range(6):
        if good.sum() < ncoeff:
            return coeff, float("nan")
        coeff = np.polynomial.legendre.legfit(x[good], positions[good], ncoeff - 1)
        fitted = np.polynomial.legendre.legval(x[good], coeff)
        residual = positions[good] - fitted
        scatter = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if scatter <= 0:
            break
        new_good = good.copy()
        new_good[good] = np.abs(residual) <= sigrej * scatter
        if np.array_equal(new_good, good):
            break
        good = new_good

    fitted = np.polynomial.legendre.legval(x[good], coeff)
    rms = float(np.sqrt(np.mean((positions[good] - fitted) ** 2)))
    return coeff, rms


def evaluate_trace(
    coeff: np.ndarray,
    xmid: float,
    xscale: float,
    rows: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted object trace at the given dispersion rows."""
    x = (np.asarray(rows, dtype=float) - xmid) / xscale
    return np.polynomial.legendre.legval(x, np.asarray(coeff, dtype=float))


__all__ = [
    "ObjectTrace",
    "collapse_profile",
    "evaluate_trace",
    "find_profile_peaks",
    "fit_trace",
    "profile_snr",
    "rectify_order",
    "trace_coordinate",
    "trace_object",
]
