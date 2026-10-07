"""Pure Step Engines for spectrophotometric flux calibration.

These functions convert detector counts into physical flux density by deriving a
per-order sensitivity function from an observed standard star. The sensitivity
is expressed as the spectroscopic zeropoint ``ZP`` of PypeIt's flux-calibration
formalism: the ``S_lam`` factor that converts a count rate per Angstrom
``N_lam`` into ``F_lam`` in units of ``1e-17 erg/s/cm^2/Angstrom`` is

    ``S_lam = 10**(-0.4 * (ZP - ZP_UNIT_CONST)) / wave**2``

and the zeropoint itself is recovered from a standard star of known flux as

    ``ZP = -2.5 * log10(wave**2 * telluric * F_std / N_lam) + ZP_UNIT_CONST``.

Everything here operates on plain NumPy arrays (ADR 0003) and is used by the
``flux_calibrate`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy import constants as const

from onfire.core.polyfit import robust_legendre_fit

PYPEIT_FLUX_SCALE = 1e-17
"""Flux unit of the calibrated spectra in erg/s/cm^2/Angstrom."""


def _zp_unit_const() -> float:
    """Unit-conversion constant anchoring the spectroscopic zeropoint."""
    flux_density = (u.AA**2 / const.c) * (
        PYPEIT_FLUX_SCALE * u.erg / u.s / u.cm**2 / u.AA
    )
    return float(
        -2.5
        * np.log10((flux_density.to("Jy") / (3631 * u.Jy)).value)
    )


ZP_UNIT_CONST = _zp_unit_const()

_AB_ZP_MIN = 5.0
_AB_ZP_MAX = 30.0


def delta_wave(wave: np.ndarray) -> np.ndarray:
    """Return the (unsigned) wavelength sampling of a 1D dispersion grid."""
    wave = np.asarray(wave, dtype=float)
    if wave.size < 2:
        return np.ones_like(wave)
    return np.abs(np.gradient(wave))


def _extinction_factor(
    extinction: object, wave: np.ndarray, airmass: float
) -> np.ndarray:
    """Return the multiplicative extinction correction ``10**(0.4*k*airmass)``."""
    if callable(extinction):
        coefficients = np.asarray(extinction(wave), dtype=float)
    else:
        coefficients = np.asarray(extinction, dtype=float)
        if coefficients.shape != wave.shape:
            coefficients = np.interp(wave, np.linspace(wave.min(), wave.max(), coefficients.size), coefficients)
    return np.power(10.0, 0.4 * coefficients * float(airmass))


def counts_to_nlam(
    wave: np.ndarray,
    counts: np.ndarray,
    exptime: float,
    *,
    counts_ivar: np.ndarray | None = None,
    sampling: np.ndarray | None = None,
    extinction: object | None = None,
    airmass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert counts to a count rate per Angstrom ``N_lam``.

    Args:
        wave: Wavelength grid in Angstroms, shape ``(nspec,)``.
        counts: Observed counts, shape ``(nspec,)``.
        exptime: Exposure time in seconds.
        counts_ivar: Optional inverse variance of ``counts``.
        sampling: Optional wavelength sampling ``dlambda``; defaults to the
            grid gradient.
        extinction: Optional atmospheric extinction coefficient array or a
            callable ``k(wave)`` returned in magnitudes per airmass.
        airmass: Airmass at which the observation was taken.

    Returns:
        ``(nlam, nlam_ivar)`` where ``nlam`` is in counts/s/Angstrom and
        ``nlam_ivar`` is ``None`` when ``counts_ivar`` was not supplied.
    """
    wave = np.asarray(wave, dtype=float)
    counts = np.asarray(counts, dtype=float)
    dw = delta_wave(wave) if sampling is None else np.asarray(sampling, dtype=float)
    exptime = float(exptime)

    with np.errstate(invalid="ignore", divide="ignore"):
        nlam = counts / exptime / dw
    nlam_ivar: np.ndarray | None = None
    if counts_ivar is not None:
        counts_ivar = np.asarray(counts_ivar, dtype=float)
        nlam_ivar = dw**2 * counts_ivar * exptime**2

    if extinction is not None:
        correction = _extinction_factor(extinction, wave, airmass)
        nlam = nlam * correction
        if nlam_ivar is not None:
            nlam_ivar = nlam_ivar / correction**2
    return nlam, nlam_ivar


def nlam_to_flam(
    wave: np.ndarray,
    zeropoint: np.ndarray,
    *,
    zp_min: float = _AB_ZP_MIN,
    zp_max: float = _AB_ZP_MAX,
) -> np.ndarray:
    """Return the ``S_lam`` factor that converts ``N_lam`` into ``F_lam``."""
    wave = np.asarray(wave, dtype=float)
    zeropoint = np.asarray(zeropoint, dtype=float)
    good = (
        (wave > 1.0)
        & np.isfinite(zeropoint)
        & (zeropoint > zp_min)
        & (zeropoint < zp_max)
    )
    factor = np.zeros_like(wave)
    factor[good] = (
        np.power(10.0, -0.4 * (zeropoint[good] - ZP_UNIT_CONST))
        / np.square(wave[good])
    )
    return factor


def compute_zeropoint(
    wave: np.ndarray,
    nlam: np.ndarray,
    standard_flux: np.ndarray,
    *,
    gpm: np.ndarray | None = None,
    telluric: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spectroscopic zeropoint from a standard star.

    Args:
        wave: Wavelength grid in Angstroms.
        nlam: Observed count rate per Angstrom of the standard star.
        standard_flux: True flux of the standard star in erg/s/cm^2/Angstrom.
        gpm: Optional initial good-pixel mask.
        telluric: Optional atmospheric transmission to remove from the standard.

    Returns:
        ``(zeropoint, good)`` with the per-pixel zeropoint and its good mask.
    """
    wave = np.asarray(wave, dtype=float)
    nlam = np.asarray(nlam, dtype=float)
    standard_flux = np.asarray(standard_flux, dtype=float)
    transmission = (
        np.ones_like(wave) if telluric is None else np.asarray(telluric, dtype=float)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.square(wave) * transmission * standard_flux / nlam
        zeropoint = -2.5 * np.log10(np.where(ratio > 0, ratio, 1.0)) + ZP_UNIT_CONST

    good = (
        (wave > 1.0)
        & (nlam > 0)
        & (ratio > 0)
        & np.isfinite(standard_flux)
        & np.isfinite(zeropoint)
    )
    if gpm is not None:
        good &= np.asarray(gpm, dtype=bool)
    return zeropoint, good


@dataclass
class SensitivityFunction:
    """A smooth per-order sensitivity function (spectroscopic zeropoint).

    The zeropoint is represented as a Legendre polynomial in a normalized
    wavelength coordinate, matching the other polynomial fits in the pipeline.

    Attributes:
        wave_min: Blue edge of the fitted wavelength domain, in Angstroms.
        wave_max: Red edge of the fitted wavelength domain, in Angstroms.
        coeffs: Legendre coefficients ordered from degree 0 upward.
        order: Polynomial degree of the fit.
        rms: RMS residual of the retained zeropoint samples.
    """

    wave_min: float
    wave_max: float
    coeffs: np.ndarray
    order: int
    rms: float = field(default=float("nan"))

    def _normalized(self, wave: np.ndarray) -> np.ndarray:
        span = self.wave_max - self.wave_min
        if span == 0:
            return np.zeros_like(np.asarray(wave, dtype=float))
        return 2.0 * (np.asarray(wave, dtype=float) - self.wave_min) / span - 1.0

    def evaluate_zeropoint(self, wave: np.ndarray) -> np.ndarray:
        """Evaluate the fitted zeropoint on ``wave``."""
        return np.polynomial.legendre.legval(
            self._normalized(wave), np.asarray(self.coeffs, dtype=float)
        )

    def factor(self, wave: np.ndarray, **kwargs: float) -> np.ndarray:
        """Evaluate the ``S_lam`` conversion factor on ``wave``."""
        return nlam_to_flam(wave, self.evaluate_zeropoint(wave), **kwargs)


def fit_sensitivity_function(
    wave: np.ndarray,
    zeropoint: np.ndarray,
    gpm: np.ndarray | None = None,
    *,
    order: int = 5,
    niter: int = 8,
    sigrej: float = 3.0,
) -> SensitivityFunction:
    """Fit a smooth sensitivity function to per-pixel zeropoints.

    Args:
        wave: Wavelength grid in Angstroms.
        zeropoint: Per-pixel spectroscopic zeropoint.
        gpm: Optional good-pixel mask.
        order: Legendre polynomial degree.
        niter: Maximum robust-rejection iterations.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        The fitted :class:`SensitivityFunction`.
    """
    wave = np.asarray(wave, dtype=float)
    zeropoint = np.asarray(zeropoint, dtype=float)
    good = np.isfinite(wave) & np.isfinite(zeropoint)
    if gpm is not None:
        good &= np.asarray(gpm, dtype=bool)

    finite_wave = wave[good]
    if finite_wave.size:
        wave_min = float(finite_wave.min())
        wave_max = float(finite_wave.max())
    else:
        wave_min, wave_max = 0.0, 1.0

    fit, _used, rms = robust_legendre_fit(
        [wave],
        zeropoint,
        (int(order),),
        in_gpm=good,
        niter=int(niter),
        sigrej=float(sigrej),
    )
    return SensitivityFunction(
        wave_min=wave_min,
        wave_max=wave_max,
        coeffs=np.asarray(fit.coeffs, dtype=float),
        order=int(order),
        rms=float(rms),
    )


def derive_sensitivity(
    wave: np.ndarray,
    counts: np.ndarray,
    standard_flux: np.ndarray,
    *,
    exptime: float,
    counts_ivar: np.ndarray | None = None,
    sampling: np.ndarray | None = None,
    airmass: float = 1.0,
    extinction: object | None = None,
    gpm: np.ndarray | None = None,
    telluric: np.ndarray | None = None,
    order: int = 5,
    niter: int = 8,
    sigrej: float = 3.0,
) -> SensitivityFunction:
    """Derive a sensitivity function from an observed standard star.

    This chains :func:`counts_to_nlam`, :func:`compute_zeropoint`, and
    :func:`fit_sensitivity_function` into the single operation the
    ``flux_calibrate`` Step Engine performs per order.
    """
    nlam, _ = counts_to_nlam(
        wave,
        counts,
        exptime,
        counts_ivar=counts_ivar,
        sampling=sampling,
        extinction=extinction,
        airmass=airmass,
    )
    zeropoint, good = compute_zeropoint(
        wave, nlam, standard_flux, gpm=gpm, telluric=telluric
    )
    return fit_sensitivity_function(
        wave, zeropoint, good, order=order, niter=niter, sigrej=sigrej
    )


def apply_sensitivity(
    sensfunc: SensitivityFunction,
    wave: np.ndarray,
    counts: np.ndarray,
    *,
    exptime: float = 1.0,
    counts_var: np.ndarray | None = None,
    sampling: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Apply a sensitivity function to counts, returning flux density.

    Args:
        sensfunc: The per-order sensitivity function.
        wave: Wavelength grid in Angstroms.
        counts: Observed counts.
        exptime: Exposure time in seconds.
        counts_var: Optional variance of ``counts`` for propagation.
        sampling: Optional wavelength sampling ``dlambda``.

    Returns:
        ``(flux, flux_var, flux_ivar, mask)`` with flux in
        erg/s/cm^2/Angstrom and the variance propagated through the linear
        sensitivity and exposure-time scaling.
    """
    wave = np.asarray(wave, dtype=float)
    counts = np.asarray(counts, dtype=float)
    dw = delta_wave(wave) if sampling is None else np.asarray(sampling, dtype=float)
    exptime = float(exptime)
    nlam, _ = counts_to_nlam(
        wave, counts, exptime, sampling=dw
    )
    zeropoint = sensfunc.evaluate_zeropoint(wave)
    factor = nlam_to_flam(wave, zeropoint)
    flux = nlam * factor
    mask = np.isfinite(flux) & (factor > 0)

    if counts_var is None:
        return flux, None, None, mask

    # flux = counts * factor / (exptime * dlambda), so the variance transforms
    # by the square of that linear scaling.
    counts_var = np.asarray(counts_var, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_var = counts_var * (factor / (exptime * dw)) ** 2
        flux_ivar = np.where(flux_var > 0, 1.0 / flux_var, 0.0)
    flux = np.where(mask, flux, np.nan)
    flux_var = np.where(mask, flux_var, np.nan)
    flux_ivar = np.where(mask, flux_ivar, 0.0)
    return flux, flux_var, flux_ivar, mask


__all__ = [
    "PYPEIT_FLUX_SCALE",
    "ZP_UNIT_CONST",
    "SensitivityFunction",
    "apply_sensitivity",
    "compute_zeropoint",
    "counts_to_nlam",
    "delta_wave",
    "derive_sensitivity",
    "fit_sensitivity_function",
    "nlam_to_flam",
]
