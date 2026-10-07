"""Pure Step Engines for 2D sky background modeling.

The functions here build the tilted line coordinate across an echelle order and
fit the sky background with a tensor-product basis: a one-dimensional B-spline
along the tilted spectral coordinate times a Legendre polynomial along the
fractional spatial coordinate. The fitted surface is evaluated across the order
and subtracted. They operate only on plain NumPy arrays (ADR 0003) and are used
by the ``subtract_sky`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onfire.core.polyfit import design_matrix, robust_matrix_fit
from onfire.core.tilts import evaluate_tilt_surface


def _bspline_design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate all B-spline basis functions at ``x`` via Cox-de Boor recursion.

    Returns:
        A ``(len(x), n_basis)`` matrix with ``n_basis = len(knots) - degree - 1``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n_total = len(knots) - 1
    if n_total <= degree:
        return np.zeros((x.size, 0))
    basis = np.zeros((n_total, x.size))
    for i in range(n_total):
        basis[i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(float)
    for k in range(1, degree + 1):
        n_cur = n_total - k
        nxt = np.zeros((n_cur, x.size))
        for i in range(n_cur):
            left_den = knots[i + k] - knots[i]
            right_den = knots[i + k + 1] - knots[i + 1]
            a = (x - knots[i]) / left_den if left_den > 0 else 0.0
            b = (knots[i + k + 1] - x) / right_den if right_den > 0 else 0.0
            nxt[i] = a * basis[i] + b * basis[i + 1]
        basis = nxt
    return basis.T


@dataclass(frozen=True)
class BSplineBasis:
    """A uniform B-spline basis defined by its knot vector and degree."""

    knots: np.ndarray
    degree: int

    @property
    def n_basis(self) -> int:
        return int(len(self.knots) - self.degree - 1)

    @classmethod
    def uniform(
        cls,
        xmin: float,
        xmax: float,
        bspacing: float,
        degree: int = 3,
    ) -> "BSplineBasis":
        """Build a uniform basis covering ``[xmin, xmax]`` with breakpoint spacing."""
        bspacing = max(float(bspacing), 1e-6)
        xmin = float(xmin)
        xmax = float(xmax)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        t0 = xmin - degree * bspacing
        t1 = xmax + degree * bspacing
        nknots = int(np.ceil((t1 - t0) / bspacing)) + 1
        knots = t0 + bspacing * np.arange(nknots, dtype=float)
        return cls(knots=knots, degree=int(degree))

    def design(self, x: np.ndarray) -> np.ndarray:
        """Return the B-spline design matrix at coordinates ``x``."""
        return _bspline_design(x, self.knots, self.degree)


@dataclass
class SkyFit:
    """A fitted 2D sky surface for one echelle order.

    The model is a tensor product of a B-spline along the tilted spectral
    coordinate and a Legendre polynomial along the fractional spatial
    coordinate.
    """

    basis: BSplineBasis
    npoly: int
    poly_domains: list[tuple[float, float]]
    coeffs: np.ndarray
    rms: float
    npoints: int

    def evaluate(self, spec_coord: np.ndarray, ximg: np.ndarray) -> np.ndarray:
        """Evaluate the sky surface at (tilted spectral, spatial) coordinates."""
        spec_coord = np.asarray(spec_coord, dtype=float).ravel()
        ximg = np.clip(np.asarray(ximg, dtype=float).ravel(), 0.0, 1.0)
        bspline = self.basis.design(spec_coord)
        poly = design_matrix([ximg], (self.npoly,), self.poly_domains)
        design = (bspline[:, :, None] * poly[:, None, :]).reshape(spec_coord.size, -1)
        return design @ np.asarray(self.coeffs, dtype=float)


def fit_sky_surface(
    spec_coord: np.ndarray,
    ximg: np.ndarray,
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    bspacing: float = 30.0,
    degree: int = 3,
    npoly: int = 2,
    sigrej: float = 3.0,
    niter: int = 8,
) -> SkyFit:
    """Fit a 2D B-spline sky surface to the background pixels of one order.

    Args:
        spec_coord: Tilted spectral coordinate of each pixel.
        ximg: Fractional spatial coordinate (0 at the left edge, 1 at the right).
        values: Sky counts to fit.
        weights: Optional per-pixel fit weights, e.g. ``sqrt(ivar)``.
        bspacing: B-spline breakpoint spacing, in pixels.
        degree: B-spline degree (3 = cubic).
        npoly: Legendre polynomial degree along the spatial axis.
        sigrej: Rejection threshold in robust sigma.
        niter: Maximum number of rejection iterations.

    Returns:
        The fitted :class:`SkyFit`.
    """
    spec_coord = np.asarray(spec_coord, dtype=float).ravel()
    ximg = np.asarray(ximg, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    weights = None if weights is None else np.asarray(weights, dtype=float).ravel()

    finite = np.isfinite(spec_coord) & np.isfinite(ximg) & np.isfinite(values)
    if weights is not None:
        finite &= np.isfinite(weights) & (weights > 0)
    min_points = (degree + 1) + (npoly + 1) + 1
    domains = [(0.0, 1.0)]
    if finite.sum() < min_points:
        return SkyFit(
            basis=BSplineBasis.uniform(0.0, 1.0, bspacing, degree),
            npoly=npoly,
            poly_domains=domains,
            coeffs=np.zeros(1),
            rms=float("nan"),
            npoints=int(finite.sum()),
        )

    basis = BSplineBasis.uniform(
        spec_coord[finite].min(), spec_coord[finite].max(), bspacing, degree
    )
    xc = np.clip(ximg[finite], 0.0, 1.0)
    bspline = basis.design(spec_coord[finite])
    poly = design_matrix([xc], (npoly,), domains)
    design = (bspline[:, :, None] * poly[:, None, :]).reshape(finite.sum(), -1)
    fit_weights = weights[finite] if weights is not None else np.ones(finite.sum())
    coeffs, _, rms = robust_matrix_fit(
        design, values[finite], weights=fit_weights, niter=niter, sigrej=sigrej
    )
    return SkyFit(
        basis=basis,
        npoly=npoly,
        poly_domains=domains,
        coeffs=coeffs,
        rms=rms,
        npoints=int(finite.sum()),
    )


def tilted_coordinates(
    nspec: int,
    nspat: int,
    left: np.ndarray,
    right: np.ndarray,
    fit,
    *,
    spat_scale: float,
) -> np.ndarray:
    """Build the tilted spectral coordinate across a full detector image.

    The coordinate follows the arc-line tilt: a pixel at spatial offset
    ``dspat`` from the order centre is assigned the spectral coordinate it
    would have on the order centre line. When no tilt surface is supplied the
    raw dispersion row coordinate is returned.

    Args:
        nspec: Number of dispersion pixels.
        nspat: Number of spatial pixels.
        left: Left order edge per dispersion row, in pixels.
        right: Right order edge per dispersion row, in pixels.
        fit: A :class:`~onfire.core.polyfit.PolynomialFit` tilt surface, or
            ``None`` to fall back to the row coordinate.
        spat_scale: Spatial normalization scale of the tilt surface.

    Returns:
        A ``(nspec, nspat)`` array of tilted spectral coordinates.
    """
    rows = np.arange(nspec, dtype=float)[:, None]
    row_grid = np.broadcast_to(rows, (nspec, nspat)).astype(float)
    if fit is None:
        return row_grid
    cols = np.arange(nspat, dtype=float)[None, :]
    center = 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))[:, None]
    dspat = cols - center
    delta = evaluate_tilt_surface(
        fit,
        row_grid.ravel(),
        dspat.ravel(),
        nspec=nspec,
        spat_scale=spat_scale,
    ).reshape(nspec, nspat)
    return row_grid - delta


__all__ = [
    "BSplineBasis",
    "SkyFit",
    "fit_sky_surface",
    "tilted_coordinates",
]
