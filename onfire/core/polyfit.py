"""Multi-dimensional Legendre polynomial fitting.

These routines provide the smooth polynomial surfaces used by the wavelength
(2D: pixel vs. order) and tilt (2D: spectral pixel vs. spatial offset) fits.
They operate only on plain NumPy arrays (ADR 0003) and use a tensor-product
Legendre basis with independent polynomial orders per coordinate, which is the
multidimensional generalization of the 1D fits used for order traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from astropy.stats import mad_std


def _exponents(orders: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return the tensor-product exponents for per-axis polynomial ``orders``."""
    return [tuple(exp) for exp in product(*(range(order + 1) for order in orders))]


def _normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == lo:
        return np.zeros_like(np.asarray(values, dtype=float))
    return 2.0 * (np.asarray(values, dtype=float) - lo) / (hi - lo) - 1.0


def _legendre_1d(x: np.ndarray, exponent: int, deriv: int = 0) -> np.ndarray:
    coeff = np.zeros(exponent + 1)
    coeff[exponent] = 1.0
    if deriv:
        coeff = np.polynomial.legendre.legder(coeff, deriv)
    if coeff.size == 0:
        return np.zeros_like(x)
    return np.polynomial.legendre.legval(x, coeff)


def design_matrix(
    coords: list[np.ndarray],
    orders: tuple[int, ...],
    domains: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build the tensor-product Legendre design matrix for ``coords``.

    Args:
        coords: One 1D array per independent coordinate, all the same length.
        orders: Polynomial order (degree) for each coordinate.
        domains: Per-coordinate ``(min, max)`` normalization domains. Defaults
            to the observed data range of each coordinate.

    Returns:
        A ``(npoints, nterms)`` matrix whose columns are the basis functions.
    """
    if len(coords) != len(orders):
        raise ValueError("coords and orders must have the same length")
    coords = [np.asarray(c, dtype=float).ravel() for c in coords]
    if domains is None:
        domains = [(float(c.min()), float(c.max())) for c in coords]
    basis: list[np.ndarray] = []
    for exps in _exponents(orders):
        column = np.ones(coords[0].size)
        for coord, exponent, (lo, hi) in zip(coords, exps, domains):
            column = column * _legendre_1d(_normalize(coord, lo, hi), exponent)
        basis.append(column)
    return np.column_stack(basis)


@dataclass
class PolynomialFit:
    """A fitted tensor-product Legendre polynomial surface.

    Attributes:
        coeffs: Flattened polynomial coefficients, in :func:`design_matrix`
            column order.
        orders: Per-coordinate polynomial orders.
        domains: Per-coordinate ``(min, max)`` normalization domains.
    """

    coeffs: np.ndarray
    orders: tuple[int, ...]
    domains: list[tuple[float, float]]

    @property
    def nterms(self) -> int:
        return int(self.coeffs.size)

    def evaluate(self, coords: list[np.ndarray]) -> np.ndarray:
        """Evaluate the surface at the given coordinates."""
        matrix = design_matrix(coords, self.orders, self.domains)
        return matrix @ np.asarray(self.coeffs, dtype=float)

    def evaluate_derivative(self, coords: list[np.ndarray], axis: int) -> np.ndarray:
        """Evaluate the partial derivative of the surface along ``axis``."""
        if len(coords) != len(self.orders):
            raise ValueError("coords and orders must have the same length")
        coords = [np.asarray(c, dtype=float).ravel() for c in coords]
        basis: list[np.ndarray] = []
        for exps in _exponents(self.orders):
            column = np.ones(coords[0].size)
            for index, (coord, exponent, (lo, hi)) in enumerate(
                zip(coords, exps, self.domains)
            ):
                if index != axis:
                    column = column * _legendre_1d(_normalize(coord, lo, hi), exponent)
                    continue
                if hi == lo:
                    column = column * np.zeros_like(coords[0])
                    continue
                x = _normalize(coord, lo, hi)
                column = column * _legendre_1d(x, exponent, deriv=1) * (2.0 / (hi - lo))
            basis.append(column)
        return np.column_stack(basis) @ np.asarray(self.coeffs, dtype=float)


def robust_matrix_fit(
    matrix: np.ndarray,
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    in_gpm: np.ndarray | None = None,
    niter: int = 8,
    sigrej: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a linear design matrix with iterative sigma clipping.

    This is the shared engine behind :func:`robust_legendre_fit` and the
    B-spline sky fits: it performs weighted least squares, rejects outliers
    beyond ``sigrej`` robust sigma, and iterates to convergence.

    Args:
        matrix: Design matrix of shape ``(npoints, nterms)``.
        values: Dependent values to fit.
        weights: Optional per-point weights, e.g. ``sqrt(ivar)``.
        in_gpm: Optional initial boolean good-point mask.
        niter: Maximum number of rejection iterations.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        ``(coeffs, good, rms)`` with the fitted coefficients, the final
        good-point mask, and the RMS residual of the retained points.
    """
    matrix = np.asarray(matrix, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    npoints = values.size
    if weights is None:
        weights = np.ones(npoints)
    weights = np.asarray(weights, dtype=float).ravel()
    good = np.isfinite(values) if in_gpm is None else np.asarray(in_gpm, dtype=bool).copy()

    coeffs = np.zeros(matrix.shape[1])
    for _ in range(max(1, niter)):
        use = good & np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if use.sum() <= matrix.shape[1]:
            break
        a = matrix[use] * weights[use][:, None]
        b = values[use] * weights[use]
        coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
        residual = values - matrix @ coeffs
        scatter = mad_std(residual[use]) if use.sum() > 1 else 0.0
        if scatter <= 0:
            break
        new_good = good & (np.abs(residual) <= sigrej * scatter)
        if np.array_equal(new_good, good):
            break
        good = new_good

    use = good & np.isfinite(values)
    residual = values[use] - matrix[use] @ coeffs
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    return coeffs, good, rms


def robust_legendre_fit(
    coords: list[np.ndarray],
    values: np.ndarray,
    orders: tuple[int, ...],
    *,
    domains: list[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    in_gpm: np.ndarray | None = None,
    niter: int = 8,
    sigrej: float = 3.0,
) -> tuple[PolynomialFit, np.ndarray, float]:
    """Fit a Legendre surface with iterative sigma clipping.

    Args:
        coords: One 1D array per independent coordinate.
        values: Dependent values to fit.
        orders: Polynomial order for each coordinate.
        domains: Per-coordinate normalization domains; defaults to data range.
        weights: Optional per-point weights, e.g. ``sqrt(ivar)``.
        in_gpm: Optional initial boolean good-point mask.
        niter: Maximum number of rejection iterations.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        ``(fit, good, rms)`` where ``good`` is the final good-point mask and
        ``rms`` the robust residual scatter of the retained points.
    """
    coords = [np.asarray(c, dtype=float).ravel() for c in coords]
    values = np.asarray(values, dtype=float).ravel()
    if domains is None:
        domains = [(float(c.min()), float(c.max())) for c in coords]
    matrix = design_matrix(coords, orders, domains)
    coeffs, good, rms = robust_matrix_fit(
        matrix, values, weights=weights, in_gpm=in_gpm, niter=niter, sigrej=sigrej
    )
    fit = PolynomialFit(coeffs, tuple(orders), list(domains))
    return fit, good, rms


__all__ = [
    "PolynomialFit",
    "design_matrix",
    "robust_legendre_fit",
    "robust_matrix_fit",
]
