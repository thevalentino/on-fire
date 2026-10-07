"""Pure Step Engines for telluric absorption correction.

This module removes the molecular absorption imprinted by Earth's atmosphere
from flux-calibrated spectra. Two independent engines are implemented:

* **Empirical** -- the intrinsic spectrum of an observed A0V standard star is
  divided out, leaving the smooth instrument response times the atmospheric
  transmission. After removing the smooth response, the residual transmission is
  applied to the science target (``empirical_transmission``).
* **Grid** -- the transmission is modelled from a precomputed PCA atmospheric
  grid (TelPCA). The PCA components linearly parameterize ``arcsinh`` of the
  optical depth, so a transmission estimate can be fit by linear least squares
  (``fit_telluric_pca``). When no standard star is available, the continuum and
  transmission are fit jointly to the science spectrum (``fit_telluric_grid``).

All functions operate on plain NumPy arrays (ADR 0003) and are used by the
``telluric_correct`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TRANSMISSION_FLOOR = 1e-3


@dataclass
class TelluricCorrection:
    """An atmospheric transmission estimate for one spectrum.

    Attributes:
        wave: Wavelength grid in Angstroms.
        transmission: Estimated transmission, in the range ``(0, 1]``.
        mask: Good-pixel mask for the transmission estimate.
        method: ``"empirical"`` or ``"grid"``.
        rms: RMS residual of the fit that produced the transmission.
        coeffs: PCA coefficients when ``method == "grid"``, else ``None``.
    """

    wave: np.ndarray
    transmission: np.ndarray
    mask: np.ndarray
    method: str
    rms: float = field(default=float("nan"))
    coeffs: np.ndarray | None = None

    def evaluate(self, wave: np.ndarray) -> np.ndarray:
        """Interpolate the transmission onto ``wave``, clipped to ``(0, 1]``."""
        wave = np.asarray(wave, dtype=float)
        template_wave = np.asarray(self.wave, dtype=float)
        template = np.asarray(self.transmission, dtype=float)
        good = np.isfinite(template_wave) & np.isfinite(template)
        if good.sum() < 2:
            return np.ones_like(wave)
        template_wave = template_wave[good]
        template = template[good]
        order = np.argsort(template_wave)
        evaluated = np.interp(wave, template_wave[order], template[order])
        return np.clip(evaluated, _TRANSMISSION_FLOOR, 1.0)


def _robust_log_continuum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    gpm: np.ndarray,
    niter: int = 6,
    sigrej: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a polynomial to ``log(y)`` versus normalized ``x`` with rejection."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.asarray(gpm, dtype=bool) & np.isfinite(x) & np.isfinite(y) & (y > 0)
    if good.sum() <= order + 1:
        return np.full_like(x, np.nan), good

    logy = np.log(y)
    design = np.vander(x, int(order) + 1, increasing=True)
    coeffs = np.zeros(design.shape[1])
    for _ in range(max(1, int(niter))):
        use = good
        if use.sum() <= design.shape[1]:
            break
        coeffs, *_ = np.linalg.lstsq(design[use], logy[use], rcond=None)
        residual = logy - design @ coeffs
        scatter = 1.4826 * float(np.median(np.abs(residual[use] - np.median(residual[use]))))
        if scatter <= 0:
            break
        new_good = good & (np.abs(residual) <= float(sigrej) * scatter)
        if np.array_equal(new_good, good):
            break
        good = new_good

    return np.exp(design @ coeffs), good


def _normalized_log_wave(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    logw = np.log10(np.clip(wave, 1.0, None))
    span = logw.max() - logw.min()
    if span == 0:
        return np.zeros_like(logw)
    return 2.0 * (logw - logw.min()) / span - 1.0


def empirical_transmission(
    wave: np.ndarray,
    standard_flux: np.ndarray,
    a0v_flux: np.ndarray,
    *,
    continuum_order: int = 6,
    gpm: np.ndarray | None = None,
    floor: float = _TRANSMISSION_FLOOR,
) -> TelluricCorrection:
    """Derive atmospheric transmission from an observed A0V standard star.

    The ratio of the observed standard spectrum to its intrinsic A0V
    photosphere is the product of the smooth instrument response and the
    atmospheric transmission. A robust smooth continuum is divided out to
    isolate the transmission, so the absolute scaling, exposure time, and
    sampling of the observed spectrum drop out.

    Args:
        wave: Wavelength grid in Angstroms.
        standard_flux: Observed standard-star spectrum (counts or flux).
        a0v_flux: Intrinsic A0V template flux on the same wavelength grid.
        continuum_order: Polynomial order of the smooth response model.
        gpm: Optional good-pixel mask.
        floor: Lower clip applied to the transmission.

    Returns:
        The empirical :class:`TelluricCorrection`.
    """
    wave = np.asarray(wave, dtype=float)
    standard_flux = np.asarray(standard_flux, dtype=float)
    a0v_flux = np.asarray(a0v_flux, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = standard_flux / a0v_flux

    good = (
        np.isfinite(ratio)
        & (ratio > 0)
        & np.isfinite(wave)
        & np.isfinite(a0v_flux)
        & (a0v_flux > 0)
    )
    if gpm is not None:
        good &= np.asarray(gpm, dtype=bool)

    continuum, _ = _robust_log_continuum(
        _normalized_log_wave(wave), ratio, order=int(continuum_order), gpm=good
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        transmission = ratio / continuum
    finite = np.isfinite(transmission)
    clipped = np.clip(np.where(finite, transmission, 1.0), floor, 1.0)
    residual = (transmission - clipped)[finite & good]
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    return TelluricCorrection(
        wave=wave,
        transmission=clipped,
        mask=good & finite,
        method="empirical",
        rms=rms,
    )


def pca_transmission(
    wave: np.ndarray,
    components: np.ndarray,
    coeffs: np.ndarray,
    *,
    floor: float = _TRANSMISSION_FLOOR,
) -> np.ndarray:
    """Evaluate a PCA atmospheric model as a transmission spectrum.

    The components linearly parameterize ``arcsinh(tau)`` where ``tau`` is the
    optical depth. Component row 0 is the model mean; ``coeffs`` scales
    components 1..n.

    Args:
        wave: Wavelength grid (used only for shape and validation).
        components: PCA component array of shape ``(ncomp, nwave)``.
        coeffs: PCA coefficients of length ``ncomp - 1`` or fewer.
        floor: Lower clip applied to the transmission.

    Returns:
        The model transmission on the component wavelength grid.
    """
    components = np.asarray(components, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float).ravel()
    ncomp = min(coeffs.size + 1, components.shape[0])
    design = np.concatenate([[1.0], coeffs[: ncomp - 1]])
    asinh_tau = design @ components[:ncomp]
    tau = np.clip(np.sinh(asinh_tau), 0.0, None)
    return np.clip(np.exp(-tau), floor, 1.0)


def _fit_pca_coeffs(
    transmission: np.ndarray,
    components: np.ndarray,
    *,
    ncoeff: int,
    gpm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit PCA coefficients to a transmission estimate by linearization."""
    transmission = np.asarray(transmission, dtype=float)
    good = (
        np.asarray(gpm, dtype=bool)
        & np.isfinite(transmission)
        & (transmission > 0)
        & (transmission <= 1.0)
    )
    ncoeff = max(1, min(int(ncoeff), components.shape[0] - 1))
    coeffs = np.zeros(ncoeff)
    if good.sum() <= ncoeff:
        return coeffs, good, float("nan")

    tau = -np.log(np.clip(transmission, _TRANSMISSION_FLOOR, 1.0))
    target = np.arcsinh(tau) - components[0]
    matrix = components[1 : ncoeff + 1].T
    coeffs, *_ = np.linalg.lstsq(matrix[good], target[good], rcond=None)
    model = pca_transmission(transmission, components, coeffs)
    residual = (transmission - model)[good]
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    return coeffs, good, rms


def fit_telluric_pca(
    wave: np.ndarray,
    transmission: np.ndarray,
    components: np.ndarray,
    *,
    ncoeff: int | None = None,
    gpm: np.ndarray | None = None,
) -> TelluricCorrection:
    """Fit a TelPCA atmospheric model to a transmission estimate.

    Args:
        wave: Wavelength grid in Angstroms.
        transmission: Observed transmission estimate, in ``(0, 1]``.
        components: PCA component array of shape ``(ncomp, nwave)``.
        ncoeff: Number of PCA components to fit; defaults to all available.
        gpm: Optional good-pixel mask.

    Returns:
        A grid :class:`TelluricCorrection` carrying the fitted coefficients.
    """
    wave = np.asarray(wave, dtype=float)
    components = np.asarray(components, dtype=float)
    good = np.isfinite(wave) & np.ones_like(wave, dtype=bool)
    if gpm is not None:
        good &= np.asarray(gpm, dtype=bool)
    coeffs, used, rms = _fit_pca_coeffs(
        transmission,
        components,
        ncoeff=components.shape[0] - 1 if ncoeff is None else int(ncoeff),
        gpm=good,
    )
    model = pca_transmission(wave, components, coeffs)
    return TelluricCorrection(
        wave=wave,
        transmission=model,
        mask=used,
        method="grid",
        rms=rms,
        coeffs=coeffs,
    )


def fit_telluric_grid(
    wave: np.ndarray,
    flux: np.ndarray,
    components: np.ndarray,
    *,
    object_order: int = 4,
    ncoeff: int | None = None,
    gpm: np.ndarray | None = None,
    niter: int = 4,
    absorption_threshold: float = 0.9,
) -> TelluricCorrection:
    """Jointly fit an object continuum and a TelPCA transmission to a spectrum.

    Used when no standard star is available: the smooth object continuum and the
    atmospheric transmission are fit together. Each iteration fits a robust
    polynomial continuum to ``flux / transmission``, then refits the PCA
    coefficients to ``flux / continuum``.

    Args:
        wave: Wavelength grid in Angstroms.
        flux: Flux-calibrated science spectrum.
        components: PCA component array of shape ``(ncomp, nwave)``.
        object_order: Polynomial order of the object continuum model.
        ncoeff: Number of PCA components to fit.
        gpm: Optional good-pixel mask.
        niter: Number of continuum/transmission refinement iterations.
        absorption_threshold: Transmission below which pixels are excluded from
            the continuum fit, so absorption features do not bias it.

    Returns:
        The grid :class:`TelluricCorrection`.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    components = np.asarray(components, dtype=float)
    ncoeff = components.shape[0] - 1 if ncoeff is None else int(ncoeff)

    base = np.isfinite(wave) & np.isfinite(flux) & (flux > 0)
    if gpm is not None:
        base &= np.asarray(gpm, dtype=bool)

    x = _normalized_log_wave(wave)
    transmission = np.ones_like(wave)
    coeffs = np.zeros(max(1, ncoeff))
    rms = float("nan")
    mask = base.copy()
    for iteration in range(max(1, int(niter))):
        with np.errstate(invalid="ignore", divide="ignore"):
            corrected = flux / transmission
        if iteration == 0:
            continuum_gpm = base
        else:
            continuum_gpm = base & (transmission >= float(absorption_threshold))
        continuum, _ = _robust_log_continuum(
            x, corrected, order=int(object_order), gpm=continuum_gpm
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            observed = flux / continuum
        coeffs, mask, rms = _fit_pca_coeffs(
            observed, components, ncoeff=ncoeff, gpm=base
        )
        transmission = pca_transmission(wave, components, coeffs)

    return TelluricCorrection(
        wave=wave,
        transmission=transmission,
        mask=mask,
        method="grid",
        rms=rms,
        coeffs=coeffs,
    )


def apply_telluric(
    wave: np.ndarray,
    flux: np.ndarray,
    correction: TelluricCorrection,
    *,
    variance: np.ndarray | None = None,
    floor: float = _TRANSMISSION_FLOOR,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Divide a spectrum by an atmospheric transmission estimate.

    Args:
        wave: Wavelength grid in Angstroms.
        flux: Flux density to correct.
        correction: The transmission estimate (empirical or grid).
        variance: Optional variance of ``flux`` for propagation.
        floor: Lower clip applied to the transmission before division.

    Returns:
        ``(corrected_flux, corrected_var, corrected_ivar, mask)``. The variance
        transforms as ``var / T**2`` and the inverse variance as ``ivar * T**2``.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    transmission = np.clip(correction.evaluate(wave), floor, 1.0)
    mask = np.isfinite(flux) & (transmission > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = flux / transmission
    corrected = np.where(mask, corrected, np.nan)

    if variance is None:
        return corrected, None, None, mask

    variance = np.asarray(variance, dtype=float)
    corrected_var = variance / transmission**2
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected_ivar = np.where(corrected_var > 0, 1.0 / corrected_var, 0.0)
    corrected_var = np.where(mask, corrected_var, np.nan)
    corrected_ivar = np.where(mask, corrected_ivar, 0.0)
    return corrected, corrected_var, corrected_ivar, mask


__all__ = [
    "TelluricCorrection",
    "apply_telluric",
    "empirical_transmission",
    "fit_telluric_grid",
    "fit_telluric_pca",
    "pca_transmission",
]
