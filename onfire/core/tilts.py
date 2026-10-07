"""Pure Step Engines for 2D spectral line tilt modeling.

The functions here trace how the spectral position of an arc line changes
across the spatial direction of an echelle order, and fit 2D Legendre
surfaces describing the spatial-spectral distortion. They operate only on
plain NumPy arrays (ADR 0003) and are used by the ``fit_tilts`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onfire.core.polyfit import PolynomialFit, robust_legendre_fit


@dataclass
class TiltMeasurements:
    """Traced tilt measurements for one echelle order.

    Attributes:
        spec_pixel: Spectral pixel position of each measured line centre.
        dspat: Spatial offset from the order centre, in pixels.
        delta: Spectral-position offset from the line centre, in pixels.
        line_spec: Line centre used as the ``delta`` reference.
        line_index: Index of the source line for each measurement.
    """

    spec_pixel: np.ndarray
    dspat: np.ndarray
    delta: np.ndarray
    line_spec: np.ndarray
    line_index: np.ndarray

    @property
    def npoints(self) -> int:
        return int(self.spec_pixel.size)


@dataclass
class TiltFit:
    """A fitted 2D tilt surface for one echelle order.

    Attributes:
        coeffs: Legendre coefficients of the normalized spectral offset.
        spec_order: Polynomial order along the spectral axis.
        spat_order: Polynomial order along the spatial axis.
        spec_scale: Spectral normalization scale (``nspec - 1``).
        spat_scale: Spatial normalization scale (half-width of the order).
        rms: RMS of the surface fit, in pixels.
        nlines: Number of traced lines contributing.
        tilt_angle: Tilt angle at the order centre, in radians.
        masked: Whether the fit failed and the order should be masked.
    """

    coeffs: np.ndarray
    spec_order: int
    spat_order: int
    spec_scale: float
    spat_scale: float
    rms: float
    nlines: int
    tilt_angle: float
    masked: bool = False


def _centroid(window: np.ndarray, guess: float, maxshift: float) -> float:
    lo = int(max(0, np.floor(guess - maxshift)))
    hi = int(min(window.size, np.ceil(guess + maxshift) + 1))
    if hi <= lo:
        return guess
    segment = np.clip(window[lo:hi], 0.0, None)
    total = segment.sum()
    if total <= 0:
        return guess
    columns = np.arange(lo, hi, dtype=float)
    return float((columns * segment).sum() / total)


def trace_line_tilt(
    arcimg: np.ndarray,
    line_spec: float,
    center_spat: float,
    *,
    half_width: float,
    maxshift: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one arc line's spectral position across the order.

    Args:
        arcimg: 2D arc image with shape ``(nspec, nspat)``.
        line_spec: Spectral pixel position of the line at the order centre.
        center_spat: Spatial pixel position of the order centre.
        half_width: Half-width of the spatial trace window, in pixels.
        maxshift: Maximum spectral shift between adjacent spatial samples.

    Returns:
        ``(dspat, positions)`` arrays of spatial offsets and the measured
        spectral positions.
    """
    arcimg = np.asarray(arcimg, dtype=float)
    nspat = arcimg.shape[1]
    center = int(round(center_spat))
    lo_spat = max(0, int(center_spat - half_width))
    hi_spat = min(nspat, int(center_spat + half_width) + 1)

    dspat: list[float] = []
    positions: list[float] = []
    guess = line_spec
    for column in range(center, hi_spat):
        guess = _centroid(arcimg[:, column], guess, maxshift)
        dspat.append(column - center_spat)
        positions.append(guess)
    guess = line_spec
    for column in range(center - 1, lo_spat - 1, -1):
        guess = _centroid(arcimg[:, column], guess, maxshift)
        dspat.append(column - center_spat)
        positions.append(guess)

    order = np.argsort(dspat)
    return np.asarray(dspat, dtype=float)[order], np.asarray(positions, dtype=float)[order]


def trace_tilts(
    arcimg: np.ndarray,
    line_specs: np.ndarray,
    center_spat: float,
    *,
    half_width: float,
    fwhm: float = 4.0,
    maxshift: float = 3.0,
) -> TiltMeasurements:
    """Trace several arc lines across one echelle order.

    Args:
        arcimg: 2D arc image with shape ``(nspec, nspat)``.
        line_specs: Spectral pixel positions of the lines to trace.
        center_spat: Spatial pixel position of the order centre.
        half_width: Half-width of the spatial trace window, in pixels.
        fwhm: Line FWHM, used to discard lines that drift too far.
        maxshift: Maximum spectral shift between adjacent spatial samples.

    Returns:
        The concatenated :class:`TiltMeasurements`.
    """
    spec_pixel: list[float] = []
    dspat: list[float] = []
    delta: list[float] = []
    line_spec: list[float] = []
    line_index: list[int] = []
    for index, position in enumerate(np.asarray(line_specs, dtype=float)):
        offsets, positions = trace_line_tilt(
            arcimg,
            position,
            center_spat,
            half_width=half_width,
            maxshift=maxshift,
        )
        if positions.size < 3:
            continue
        if np.nanmax(np.abs(positions - position)) > 3.0 * fwhm + 1.0:
            continue
        spec_pixel.extend(positions.tolist())
        dspat.extend(offsets.tolist())
        delta.extend((positions - position).tolist())
        line_spec.extend([float(position)] * positions.size)
        line_index.extend([index] * positions.size)
    return TiltMeasurements(
        spec_pixel=np.asarray(spec_pixel, dtype=float),
        dspat=np.asarray(dspat, dtype=float),
        delta=np.asarray(delta, dtype=float),
        line_spec=np.asarray(line_spec, dtype=float),
        line_index=np.asarray(line_index, dtype=int),
    )


def _spatial_scale(half_width: float) -> float:
    return float(max(half_width, 1.0))


def fit_tilt_surface(
    measurements: TiltMeasurements,
    *,
    nspec: int,
    spat_scale: float,
    spec_order: int = 3,
    spat_order: int = 4,
    sigrej: float = 3.0,
) -> TiltFit:
    """Fit the 2D spectral-offset surface for one order.

    Args:
        measurements: Traced tilt measurements.
        nspec: Number of detector pixels along dispersion.
        spat_scale: Spatial normalization scale (order half-width).
        spec_order: Polynomial order along the spectral axis.
        spat_order: Polynomial order along the spatial axis.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        The fitted :class:`TiltFit`.
    """
    xnspecmin1 = float(max(1, nspec - 1))
    spat_scale = _spatial_scale(spat_scale)
    if measurements.npoints < 4:
        return _masked_tilt(nspec, spec_order, spat_order, spat_scale)

    spec_coord = measurements.spec_pixel / xnspecmin1
    spat_coord = measurements.dspat / spat_scale
    delta_norm = measurements.delta / xnspecmin1
    fit, good, _ = robust_legendre_fit(
        [spec_coord, spat_coord],
        delta_norm,
        (max(0, spec_order - 1), max(0, spat_order - 1)),
        domains=[(0.0, 1.0), (-1.0, 1.0)],
        sigrej=sigrej,
    )
    residual = (measurements.delta - xnspecmin1 * fit.evaluate([spec_coord, spat_coord]))[good]
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    angle = tilt_angle(
        fit, nspec=nspec, spat_scale=spat_scale, spec_norm=0.5, spat_norm=0.0
    )
    return TiltFit(
        coeffs=np.asarray(fit.coeffs, dtype=float),
        spec_order=spec_order,
        spat_order=spat_order,
        spec_scale=xnspecmin1,
        spat_scale=spat_scale,
        rms=rms,
        nlines=int(np.unique(measurements.line_index).size),
        tilt_angle=angle,
        masked=False,
    )


def _masked_tilt(
    nspec: int, spec_order: int, spat_order: int, spat_scale: float
) -> TiltFit:
    nterms = spec_order * spat_order
    return TiltFit(
        coeffs=np.full(nterms, np.nan),
        spec_order=spec_order,
        spat_order=spat_order,
        spec_scale=float(max(1, nspec - 1)),
        spat_scale=spat_scale,
        rms=float("nan"),
        nlines=0,
        tilt_angle=float("nan"),
        masked=True,
    )


def build_tilt_fit(tilt: TiltFit) -> PolynomialFit:
    """Rebuild the :class:`PolynomialFit` underlying a :class:`TiltFit`."""
    return PolynomialFit(
        np.asarray(tilt.coeffs, dtype=float),
        (max(0, tilt.spec_order - 1), max(0, tilt.spat_order - 1)),
        [(0.0, 1.0), (-1.0, 1.0)],
    )


def tilt_angle(
    fit: PolynomialFit,
    *,
    nspec: int,
    spat_scale: float,
    spec_norm: float = 0.5,
    spat_norm: float = 0.0,
) -> float:
    """Derive the spectral-line tilt angle at a position in the order.

    The tilt angle is the angle between a spectral line and the spatial axis,
    whose tangent is the local derivative of the spectral position with
    respect to the spatial position.

    Args:
        fit: The normalized spectral-offset surface.
        nspec: Number of detector pixels along dispersion.
        spat_scale: Spatial normalization scale (order half-width).
        spec_norm: Normalized spectral coordinate at which to evaluate.
        spat_norm: Normalized spatial coordinate at which to evaluate.

    Returns:
        The tilt angle in radians.
    """
    xnspecmin1 = float(max(1, nspec - 1))
    spat_scale = _spatial_scale(spat_scale)
    d_norm = fit.evaluate_derivative(
        [np.array([spec_norm]), np.array([spat_norm])], axis=1
    )[0]
    slope = xnspecmin1 / spat_scale * float(d_norm)
    return float(np.arctan(slope))


def evaluate_tilt_surface(
    fit: PolynomialFit,
    spec_pixel: np.ndarray,
    dspat: np.ndarray,
    *,
    nspec: int,
    spat_scale: float,
) -> np.ndarray:
    """Evaluate a tilt surface, returning spectral-position offsets in pixels."""
    xnspecmin1 = float(max(1, nspec - 1))
    spat_scale = _spatial_scale(spat_scale)
    spec_coord = np.asarray(spec_pixel, dtype=float) / xnspecmin1
    spat_coord = np.asarray(dspat, dtype=float) / spat_scale
    return xnspecmin1 * fit.evaluate([spec_coord, spat_coord])


def regularize_tilt_fit(
    measurements: TiltMeasurements,
    prior: PolynomialFit,
    *,
    nspec: int,
    spat_scale: float,
    spec_order: int = 3,
    spat_order: int = 4,
    correction_order: int = 1,
    sigrej: float = 3.0,
) -> TiltFit:
    """Refine one order's tilt surface using a global cross-order prior.

    A smooth low-order correction to the global surface is fit to the traced
    residuals, so a faint or noisy order inherits the instrument distortion
    rather than being dropped.

    Args:
        measurements: Traced tilt measurements for the target order.
        prior: Global tilt surface across all orders.
        nspec: Number of detector pixels along dispersion.
        spat_scale: Spatial normalization scale (order half-width).
        spec_order: Stored spectral polynomial order.
        spat_order: Stored spatial polynomial order.
        correction_order: Polynomial order of the correction surface.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        The regularized :class:`TiltFit`.
    """
    xnspecmin1 = float(max(1, nspec - 1))
    spat_scale = _spatial_scale(spat_scale)
    if measurements.npoints < 4:
        return _tilt_from_prior(prior, nspec, spec_order, spat_order, spat_scale)

    spec_coord = measurements.spec_pixel / xnspecmin1
    spat_coord = measurements.dspat / spat_scale
    delta_norm = measurements.delta / xnspecmin1
    prior_delta = prior.evaluate([spec_coord, spat_coord])
    correction, _, _ = robust_legendre_fit(
        [spec_coord, spat_coord],
        delta_norm - prior_delta,
        (max(0, correction_order - 1), max(0, correction_order - 1)),
        domains=[(0.0, 1.0), (-1.0, 1.0)],
        sigrej=sigrej,
    )
    model = _combine_surface(prior, correction)
    residual = (
        measurements.delta
        - xnspecmin1 * model.evaluate([spec_coord, spat_coord])
    )
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    angle = tilt_angle(
        model, nspec=nspec, spat_scale=spat_scale, spec_norm=0.5, spat_norm=0.0
    )
    return TiltFit(
        coeffs=np.asarray(model.coeffs, dtype=float),
        spec_order=spec_order,
        spat_order=spat_order,
        spec_scale=xnspecmin1,
        spat_scale=spat_scale,
        rms=rms,
        nlines=int(np.unique(measurements.line_index).size),
        tilt_angle=angle,
        masked=False,
    )


def _combine_surface(prior: PolynomialFit, correction: PolynomialFit) -> PolynomialFit:
    """Evaluate ``prior + correction`` onto one tensor-product basis.

    Both surfaces share the same 2D Legendre domains; the correction (a lower
    order surface) is expanded into the prior basis by evaluating it on the
    shared basis grid.  To keep the stored coefficient array in the prior's
    basis, the sum is re-fit to the union of both surfaces sampled on a dense
    grid.
    """
    spec_axis = np.linspace(0.0, 1.0, 32)
    spat_axis = np.linspace(-1.0, 1.0, 32)
    spec_grid, spat_grid = np.meshgrid(spec_axis, spat_axis, indexing="ij")
    total = prior.evaluate([spec_grid.ravel(), spat_grid.ravel()]) + correction.evaluate(
        [spec_grid.ravel(), spat_grid.ravel()]
    )
    fit, _, _ = robust_legendre_fit(
        [spec_grid.ravel(), spat_grid.ravel()],
        total,
        prior.orders,
        domains=prior.domains,
        niter=1,
    )
    return fit


def _tilt_from_prior(
    prior: PolynomialFit,
    nspec: int,
    spec_order: int,
    spat_order: int,
    spat_scale: float,
) -> TiltFit:
    target = _resample_surface(prior, (max(0, spec_order - 1), max(0, spat_order - 1)))
    angle = tilt_angle(
        target, nspec=nspec, spat_scale=spat_scale, spec_norm=0.5, spat_norm=0.0
    )
    return TiltFit(
        coeffs=np.asarray(target.coeffs, dtype=float),
        spec_order=spec_order,
        spat_order=spat_order,
        spec_scale=float(max(1, nspec - 1)),
        spat_scale=spat_scale,
        rms=float("nan"),
        nlines=0,
        tilt_angle=angle,
        masked=False,
    )


def _resample_surface(fit: PolynomialFit, orders: tuple[int, ...]) -> PolynomialFit:
    """Express ``fit`` on a different tensor-product Legendre basis."""
    spec_axis = np.linspace(0.0, 1.0, 24)
    spat_axis = np.linspace(-1.0, 1.0, 24)
    spec_grid, spat_grid = np.meshgrid(spec_axis, spat_axis, indexing="ij")
    values = fit.evaluate([spec_grid.ravel(), spat_grid.ravel()])
    target, _, _ = robust_legendre_fit(
        [spec_grid.ravel(), spat_grid.ravel()],
        values,
        orders,
        domains=[(0.0, 1.0), (-1.0, 1.0)],
        niter=1,
    )
    return target


def fit_global_tilt_surface(
    spec_pixel: np.ndarray,
    dspat: np.ndarray,
    delta: np.ndarray,
    *,
    nspec: int,
    spat_scale: float,
    spec_order: int = 3,
    spat_order: int = 4,
    sigrej: float = 3.0,
) -> PolynomialFit:
    """Fit a single global tilt surface pooled across all orders."""
    if np.asarray(spec_pixel).size < 4:
        return PolynomialFit(
            np.zeros(spec_order * spat_order),
            (max(0, spec_order - 1), max(0, spat_order - 1)),
            [(0.0, 1.0), (-1.0, 1.0)],
        )
    xnspecmin1 = float(max(1, nspec - 1))
    spat_scale = _spatial_scale(spat_scale)
    fit, _, _ = robust_legendre_fit(
        [np.asarray(spec_pixel, dtype=float) / xnspecmin1, np.asarray(dspat, dtype=float) / spat_scale],
        np.asarray(delta, dtype=float) / xnspecmin1,
        (max(0, spec_order - 1), max(0, spat_order - 1)),
        domains=[(0.0, 1.0), (-1.0, 1.0)],
        sigrej=sigrej,
    )
    return fit


__all__ = [
    "TiltFit",
    "TiltMeasurements",
    "build_tilt_fit",
    "evaluate_tilt_surface",
    "fit_global_tilt_surface",
    "fit_tilt_surface",
    "regularize_tilt_fit",
    "tilt_angle",
    "trace_line_tilt",
    "trace_tilts",
]
