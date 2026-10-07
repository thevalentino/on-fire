"""Pure Step Engines for order coaddition.

Adjacent echelle orders overlap slightly in wavelength. This module splices the
calibrated per-order spectra of a single object into one continuous 1D spectrum
spanning the full FIRE range, combining overlapping samples by inverse-variance
weighting onto a common wavelength grid.

Everything here operates on plain NumPy arrays (ADR 0003) and is used by the
``coadd_orders`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoaddedSpectrum:
    """A continuous 1D spectrum spliced from overlapping echelle orders.

    Attributes:
        wave: Sorted wavelength grid in Angstroms.
        flux: Coadded flux density.
        var: Coadded variance.
        ivar: Coadded inverse variance, zero where no order contributed.
        mask: Good-sample mask (``ivar > 0`` and finite flux).
        norders: Number of input order segments that were combined.
        npoints: Number of output grid samples.
    """

    wave: np.ndarray
    flux: np.ndarray
    var: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray
    norders: int = 0
    npoints: int = field(default=0)


def _grid_step(waves: list[np.ndarray]) -> float | None:
    """Return the finest per-order wavelength sampling, or ``None``."""
    steps: list[float] = []
    for wave in waves:
        wave = np.asarray(wave, dtype=float)
        finite = wave[np.isfinite(wave)]
        if finite.size < 2:
            continue
        diff = np.abs(np.diff(np.sort(finite)))
        positive = diff[diff > 0]
        if positive.size:
            steps.append(float(np.median(positive)))
    if not steps:
        return None
    return min(steps)


def coadd_order_spectra(
    waves: list[np.ndarray],
    fluxes: list[np.ndarray],
    variances: list[np.ndarray] | None = None,
    masks: list[np.ndarray] | None = None,
) -> CoaddedSpectrum:
    """Splice and inverse-variance weight overlapping order spectra.

    Args:
        waves: One wavelength grid per order, in Angstroms.
        fluxes: One flux density array per order.
        variances: Optional variance array per order. When omitted every sample
            is weighted equally.
        masks: Optional good-sample mask per order.

    Returns:
        The :class:`CoaddedSpectrum` on a common wavelength grid.

    Raises:
        ValueError: If the segment lists have differing lengths.
    """
    if not (len(waves) == len(fluxes)):
        raise ValueError("waves and fluxes must have the same number of segments")
    if variances is not None and len(variances) != len(waves):
        raise ValueError("variances must have one entry per wavelength segment")
    if masks is not None and len(masks) != len(waves):
        raise ValueError("masks must have one entry per wavelength segment")

    norders = len(waves)
    all_wave: list[np.ndarray] = []
    all_flux: list[np.ndarray] = []
    all_weight: list[np.ndarray] = []
    clean_waves: list[np.ndarray] = []
    for i, (wave, flux) in enumerate(zip(waves, fluxes)):
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        good = np.isfinite(wave) & np.isfinite(flux)
        if masks is not None:
            good &= np.asarray(masks[i], dtype=bool)
        if variances is not None:
            var = np.asarray(variances[i], dtype=float)
            positive = np.isfinite(var) & (var > 0)
            good &= positive
            weight = np.where(positive, 1.0 / np.where(positive, var, 1.0), 0.0)
        else:
            weight = np.ones_like(wave)
        if not good.any():
            continue
        clean_waves.append(wave[good])
        all_wave.append(wave[good])
        all_flux.append(flux[good])
        all_weight.append(weight[good])

    if not all_wave:
        empty = np.array([], dtype=float)
        return CoaddedSpectrum(
            wave=empty,
            flux=empty.copy(),
            var=empty.copy(),
            ivar=empty.copy(),
            mask=np.array([], dtype=bool),
            norders=norders,
            npoints=0,
        )

    step = _grid_step(clean_waves) or 1.0
    wave_flat = np.concatenate(all_wave)
    flux_flat = np.concatenate(all_flux)
    weight_flat = np.concatenate(all_weight)

    wave_min = float(wave_flat.min())
    wave_max = float(wave_flat.max())
    edges = np.arange(wave_min, wave_max + 1.5 * step, step)
    nbin = max(1, edges.size - 1)
    centers = edges[:-1] + 0.5 * step

    index = np.digitize(wave_flat, edges) - 1
    valid = (index >= 0) & (index < nbin)
    index = np.clip(index, 0, nbin - 1)

    summed_weight = np.bincount(index[valid], weights=weight_flat[valid], minlength=nbin)
    summed_flux = np.bincount(
        index[valid], weights=(flux_flat * weight_flat)[valid], minlength=nbin
    )

    occupied = summed_weight > 0
    wave_out = centers[occupied]
    ivar_out = summed_weight[occupied]
    with np.errstate(invalid="ignore", divide="ignore"):
        flux_out = summed_flux[occupied] / summed_weight[occupied]
        var_out = 1.0 / summed_weight[occupied]
    mask = np.isfinite(flux_out) & (ivar_out > 0)
    return CoaddedSpectrum(
        wave=wave_out,
        flux=flux_out,
        var=var_out,
        ivar=ivar_out,
        mask=mask,
        norders=norders,
        npoints=int(occupied.sum()),
    )


__all__ = ["CoaddedSpectrum", "coadd_order_spectra"]
