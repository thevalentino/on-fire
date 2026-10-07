"""Pure Step Engines for Galactic dust extinction correction.

This module corrects top-of-atmosphere fluxes for foreground Milky Way dust,
recovering intrinsic astrophysical values. The reddening is parameterized by the
colour excess ``E(B-V)`` and described by the Fitzpatrick (1999) extinction law
``A(lambda)/A(V) = a(x) + b(x)/R_V`` with ``x = 1e4 / lambda``.

Everything here operates on plain NumPy arrays (ADR 0003) and is used by the
``extinction_correct`` Step Engine.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

RV_DEFAULT = 3.1
"""Default total-to-selective extinction ratio ``R_V``."""

_FACTOR_SCALE = 0.4
"""Magnitude-to-flux conversion: ``flux * 10**(0.4 * A_lambda)``."""


def _f99_coefficients(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fitzpatrick (1999) ``a(x)`` and ``b(x)`` coefficients."""
    x = np.asarray(x, dtype=float)
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # Infrared: valid for 0.3 <= x <= 1.1.
    ir = x <= 1.1
    a[ir] = 0.574 * np.power(x[ir], 1.61)
    b[ir] = -0.527 * np.power(x[ir], 1.61)

    # Optical: valid for 1.1 < x <= 3.3.
    optical = (x > 1.1) & (x <= 3.3)
    y = x[optical] - 1.82
    a[optical] = (
        1.0
        + y
        * (
            0.17699
            + y * (-0.50447 + y * (-0.02427 + y * (0.72085 + y * (0.01979 + y * (-0.77530 + y * 0.32999)))) )
        )
    )
    b[optical] = y * (
        1.41338
        + y
        * (
            2.28305
            + y * (1.07233 + y * (-5.38434 + y * (-0.62251 + y * (5.30260 + y * (-2.09002)))))
        )
    )

    # Ultraviolet: valid for 3.3 < x <= 8.
    uv = (x > 3.3) & (x <= 8.0)
    xu = x[uv]
    fa = np.zeros_like(xu)
    fb = np.zeros_like(xu)
    far = xu >= 5.9
    dy = xu[far] - 5.9
    fa[far] = -0.04473 * dy**2 - 0.009779 * dy**3
    fb[far] = 0.2130 * dy**2 + 0.1207 * dy**3
    a[uv] = (
        1.752
        - 0.316 * xu
        - 0.104 / ((xu - 4.67) ** 2 + 0.341)
        + fa
    )
    b[uv] = (
        -3.090
        + 1.825 * xu
        + 1.206 / ((xu - 4.62) ** 2 + 0.263)
        + fb
    )

    # Far ultraviolet: extrapolate from the last valid coefficient at x = 8.
    beyond = x > 8.0
    if np.any(beyond):
        a8, b8 = _f99_coefficients(np.array([8.0]))
        a[beyond] = a8[0]
        b[beyond] = b8[0]

    return a, b


def fitzpatrick99(
    wave: np.ndarray,
    ebv: float = 1.0,
    rv: float = RV_DEFAULT,
) -> np.ndarray:
    """Return the F99 extinction ``A(lambda)`` in magnitudes.

    Args:
        wave: Wavelength grid in Angstroms.
        ebv: Colour excess ``E(B-V)`` in magnitudes.
        rv: Total-to-selective extinction ratio ``R_V``.

    Returns:
        The extinction ``A(lambda)`` in magnitudes on ``wave``.
    """
    wave = np.asarray(wave, dtype=float)
    x = np.where(wave > 0, 1.0e4 / np.where(wave > 0, wave, 1.0), 0.0)
    a, b = _f99_coefficients(x)
    return float(ebv) * (float(rv) * a + b)


def extinction_factor(
    wave: np.ndarray,
    ebv: float,
    rv: float = RV_DEFAULT,
) -> np.ndarray:
    """Return the multiplicative dereddening factor ``10**(0.4 * A(lambda))``."""
    return np.power(10.0, _FACTOR_SCALE * fitzpatrick99(wave, ebv=ebv, rv=rv))


def query_ebv(
    ra: float,
    dec: float,
    *,
    catalog: Table | None = None,
) -> float:
    """Return the ``E(B-V)`` at target coordinates from a local dust map.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        catalog: A local dust map with ``ra``, ``dec`` and ``ebv`` columns. When
            ``None`` no map is available and this is an error.

    Returns:
        The colour excess of the nearest catalog entry.

    Raises:
        ValueError: If no dust map was supplied.
    """
    if catalog is None:
        raise ValueError(
            "no dust map available to query E(B-V); pass 'catalog' or set the "
            "'ebv' override in the Parameter Config"
        )
    catalog_ra = np.asarray(catalog["ra"], dtype=float)
    catalog_dec = np.asarray(catalog["dec"], dtype=float)
    catalog_ebv = np.asarray(catalog["ebv"], dtype=float)
    squared = (catalog_ra - float(ra)) ** 2 + (catalog_dec - float(dec)) ** 2
    return float(catalog_ebv[int(np.argmin(squared))])


def deredden_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    ebv: float,
    *,
    rv: float = RV_DEFAULT,
    variance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Correct a spectrum for Galactic dust extinction.

    Args:
        wave: Wavelength grid in Angstroms.
        flux: Observed flux density.
        ebv: Colour excess ``E(B-V)`` in magnitudes.
        rv: Total-to-selective extinction ratio ``R_V``.
        variance: Optional variance of ``flux`` for propagation.

    Returns:
        ``(flux, var, ivar, mask)`` with the intrinsic flux and variance. The
        variance transforms as ``var * factor**2`` and the inverse variance as
        ``ivar / factor**2``.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    factor = extinction_factor(wave, ebv, rv=rv)
    mask = np.isfinite(flux) & (factor > 0)
    corrected = np.where(mask, flux * factor, np.nan)

    if variance is None:
        return corrected, None, None, mask

    variance = np.asarray(variance, dtype=float)
    corrected_var = variance * factor**2
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected_ivar = np.where(corrected_var > 0, 1.0 / corrected_var, 0.0)
    corrected_var = np.where(mask, corrected_var, np.nan)
    corrected_ivar = np.where(mask, corrected_ivar, 0.0)
    return corrected, corrected_var, corrected_ivar, mask


__all__ = [
    "RV_DEFAULT",
    "deredden_spectrum",
    "extinction_factor",
    "fitzpatrick99",
    "query_ebv",
]
