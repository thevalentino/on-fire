"""Pure Step Engines for echelle order edge tracing.

The functions here locate the left and right spatial boundaries of each FIRE
echelle order on the detector and follow them along the dispersion axis. They
operate only on plain NumPy arrays (ADR 0003) and are used by the
``trace_orders`` Step Engine.
"""

from __future__ import annotations

import numpy as np

from onfire.core.numerics import boxcar_1d


def trace_coordinate(nspec: int) -> tuple[float, float]:
    """Return the dispersion-coordinate midpoint and scale for ``nspec`` rows."""
    xmid = (nspec - 1) / 2.0
    return xmid, (xmid if xmid > 0 else 1.0)


def _search_half(index: int, centers: np.ndarray, nspat: int) -> float:
    if centers.size == 1:
        return 0.5 * nspat
    if index == 0:
        return 0.5 * (centers[1] - centers[0])
    if index == centers.size - 1:
        return 0.5 * (centers[-1] - centers[-2])
    gap = min(centers[index] - centers[index - 1], centers[index + 1] - centers[index])
    return 0.5 * gap


def detect_order_edges(
    flat: np.ndarray,
    order_spat_pos: np.ndarray,
    *,
    profile_window: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate the left and right spatial edges of each echelle order.

    The order centres are found as peaks of the dispersion-collapsed spatial
    profile, refined from the expected fractional order positions. Each left
    edge is then the strongest rising spatial gradient between the previous
    order and the centre, and each right edge the strongest falling gradient
    between the centre and the next order.

    Args:
        flat: Combined flat with shape ``(nspec, nspat)``.
        order_spat_pos: Expected fractional spatial position of each order.
        profile_window: Boxcar size used to smooth the collapsed profile.

    Returns:
        ``(left, right, centers)`` arrays of pixel positions at the reference
        row, all ordered to match ``order_spat_pos``.
    """
    flat = np.asarray(flat, dtype=float)
    if flat.ndim != 2:
        raise ValueError("flat must be 2D")
    _, nspat = flat.shape

    expected = np.asarray(order_spat_pos, dtype=float) * (nspat - 1)
    profile = boxcar_1d(np.nanmedian(flat, axis=0), profile_window)
    gradient = np.gradient(profile)

    centers = np.empty(expected.size, dtype=float)
    for i, center in enumerate(expected):
        half = _search_half(i, expected, nspat)
        lo = int(max(0, np.floor(center - half)))
        hi = int(min(nspat, np.ceil(center + half) + 1))
        if hi <= lo:
            centers[i] = center
            continue
        centers[i] = lo + float(np.argmax(profile[lo:hi]))

    left = np.empty(expected.size, dtype=float)
    right = np.empty(expected.size, dtype=float)
    for i, center in enumerate(centers):
        half = _search_half(i, centers, nspat)
        lo = int(max(0, np.floor(center - half)))
        hi = int(min(nspat, np.floor(center)))
        left[i] = lo + float(np.argmax(gradient[lo:hi])) if hi > lo else center
        lo = int(max(0, np.floor(center)))
        hi = int(min(nspat, np.ceil(center + half) + 1))
        right[i] = lo + float(np.argmin(gradient[lo:hi])) if hi > lo else center

    return left, right, centers


def _recenter(window: np.ndarray, previous: float, maxshift: float) -> float:
    lo = int(max(0, np.floor(previous - maxshift)))
    hi = int(min(window.size, np.ceil(previous + maxshift) + 1))
    if hi <= lo:
        return previous
    segment = np.clip(window[lo:hi], 0.0, None)
    total = segment.sum()
    if total <= 0:
        return previous
    columns = np.arange(lo, hi, dtype=float)
    centroid = float((columns * segment).sum() / total)
    return float(np.clip(centroid, previous - maxshift, previous + maxshift))


def trace_columns(
    gradient: np.ndarray,
    starts: np.ndarray,
    signs: np.ndarray,
    *,
    maxshift: float = 3.0,
    ref_row: int | None = None,
) -> np.ndarray:
    """Follow edge positions along the dispersion axis.

    Each edge is recentred row by row within ``maxshift`` pixels of its
    position in the adjacent row, starting from the reference row and working
    outward. ``signs`` selects rising (+1, left edges) or falling (-1, right
    edges) spatial gradients.

    Returns:
        An array of shape ``(nspec, len(starts))`` of tracked edge centroids.
    """
    gradient = np.asarray(gradient, dtype=float)
    nspec = gradient.shape[0]
    if ref_row is None:
        ref_row = nspec // 2
    cen = np.full((nspec, len(starts)), np.nan)
    for j, (start, sign) in enumerate(zip(np.atleast_1d(starts), np.atleast_1d(signs))):
        previous = _recenter(sign * gradient[ref_row], float(start), maxshift)
        cen[ref_row, j] = previous
        for row in range(ref_row + 1, nspec):
            previous = _recenter(sign * gradient[row], previous, maxshift)
            cen[row, j] = previous
        for row in range(ref_row - 1, -1, -1):
            previous = _recenter(sign * gradient[row], previous, maxshift)
            cen[row, j] = previous
    return cen


def _xcoordinate(rows: np.ndarray, xmid: float, xscale: float) -> np.ndarray:
    return (np.asarray(rows, dtype=float) - xmid) / xscale


def fit_edge_traces(
    cen: np.ndarray,
    *,
    ncoeff: int,
    ref_row: int | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Fit each tracked edge with a Legendre polynomial along dispersion.

    Args:
        cen: Edge centroid array of shape ``(nspec, n_edges)``.
        ncoeff: Number of polynomial coefficients (fit order ``ncoeff - 1``).
        ref_row: Central row used to normalize the dispersion coordinate.

    Returns:
        ``(coefficients, rms)`` lists, one entry per edge. A failed fit yields
        an all-NaN coefficient array and NaN RMS.
    """
    cen = np.asarray(cen, dtype=float)
    nspec = cen.shape[0]
    if ref_row is None:
        ref_row = (nspec - 1) // 2
    xmid, xscale = trace_coordinate(nspec)
    rows = np.arange(nspec)
    x = _xcoordinate(rows, xmid, xscale)

    coefficients: list[np.ndarray] = []
    rms: list[float] = []
    for edge in range(cen.shape[1]):
        values = cen[:, edge]
        valid = np.isfinite(values)
        if valid.sum() < ncoeff:
            coefficients.append(np.full(ncoeff, np.nan))
            rms.append(float("nan"))
            continue
        coeff = np.polynomial.legendre.legfit(x[valid], values[valid], ncoeff - 1)
        fitted = np.polynomial.legendre.legval(x[valid], coeff)
        coefficients.append(coeff)
        rms.append(float(np.sqrt(np.mean((values[valid] - fitted) ** 2))))
    return coefficients, rms


def evaluate_edge_trace(
    coeff: np.ndarray,
    xmid: float,
    xscale: float,
    rows: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted edge polynomial at the given dispersion rows."""
    x = _xcoordinate(rows, xmid, xscale)
    return np.polynomial.legendre.legval(x, np.asarray(coeff, dtype=float))
