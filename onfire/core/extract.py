"""Pure Step Engines for 1D spectral extraction.

These functions turn a sky-subtracted order image into a one-dimensional
spectrum. They implement a standard boxcar sum, a fitted spatial object
profile, and Horne (1986) optimal extraction with cosmic-ray rejection. Formal
variance is propagated throughout. They operate only on plain NumPy arrays
(ADR 0003) and are used by the ``extract_1d`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_PER_SIGMA = 2.3548200450309493


@dataclass
class BoxcarExtraction:
    """A boxcar-extracted spectrum and its propagated variance.

    Attributes:
        flux: Extracted flux per dispersion pixel.
        var: Variance of the extracted flux.
        ivar: Inverse variance of the extracted flux.
        npix: Number of good pixels that contributed to each pixel.
        mask: True where at least one good pixel contributed.
    """

    flux: np.ndarray
    var: np.ndarray
    ivar: np.ndarray
    npix: np.ndarray
    mask: np.ndarray


@dataclass
class OptimalExtraction:
    """A Horne optimal-extracted spectrum and its propagated variance.

    Attributes:
        flux: Optimally extracted flux per dispersion pixel.
        var: Variance of the optimally extracted flux.
        ivar: Inverse variance of the optimally extracted flux.
        mask: True where the extraction is usable.
        frac_use: Fraction of the normalized object profile that was used.
        chi2: Reduced chi-square of the profile model.
        cr_mask: Boolean image of pixels rejected as cosmic rays.
    """

    flux: np.ndarray
    var: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray
    frac_use: np.ndarray
    chi2: np.ndarray
    cr_mask: np.ndarray


@dataclass
class ProfileFit:
    """A fitted spatial object profile.

    The profile is one-dimensional in the pixel offset from the object trace,
    not yet normalized. ``kind`` selects the parameterization:

    * ``"gaussian"``: ``coeffs = (amplitude, centre, sigma)``.
    * ``"polynomial"``: ``coeffs = (centre, sigma, *legendre)`` evaluated in the
      normalized offset ``(offset - centre) / (3 * sigma)``.

    Attributes:
        kind: Profile parameterization name.
        coeffs: Model coefficients, interpreted according to ``kind``.
        sigma: Profile width, in pixels.
        fwhm: Profile full width at half maximum, in pixels.
        rms: RMS of the fit to the supplied profile samples.
    """

    kind: str
    coeffs: np.ndarray
    sigma: float
    fwhm: float
    rms: float

    def evaluate(self, offsets: np.ndarray) -> np.ndarray:
        """Evaluate the (unnormalized) profile at pixel offsets."""
        offsets = np.asarray(offsets, dtype=float)
        if self.kind == "polynomial":
            centre = float(self.coeffs[0])
            sigma = max(float(self.coeffs[1]), 1e-6)
            x = np.clip((offsets - centre) / (3.0 * sigma), -1.0, 1.0)
            values = np.polynomial.legendre.legval(x, self.coeffs[2:])
            return np.clip(values, 0.0, None)
        sigma = max(float(self.coeffs[2]), 1e-6)
        return np.exp(-0.5 * ((offsets - float(self.coeffs[1])) / sigma) ** 2)


def _moment_gaussian(
    offsets: np.ndarray, values: np.ndarray, peak: float
) -> tuple[float, float, float]:
    """Return a moment-estimated ``(centre, sigma, amplitude)`` Gaussian."""
    total = float(values.sum())
    if total > 0:
        centre = float((offsets * values).sum() / total)
        variance = float((values * (offsets - centre) ** 2).sum() / total)
    else:
        centre = float(offsets[np.argmax(values)])
        variance = 1.0
    sigma = max(np.sqrt(max(variance, 1.0)), 1.0)
    return centre, sigma, peak


def _fit_gaussian(offsets: np.ndarray, values: np.ndarray) -> ProfileFit:
    """Fit a Gaussian by a robust log-parabola least-squares fit."""
    peak = float(np.max(values))
    if peak <= 0:
        return _degenerate_fit(offsets, values)
    use = values > 0.05 * peak
    if use.sum() < 3:
        use = values > 0
    if use.sum() < 3:
        return _degenerate_fit(offsets, values)

    x = offsets[use]
    y = values[use]
    quadratic, linear, constant = np.polyfit(x, np.log(y), 2)
    if quadratic >= 0:
        centre, sigma, amplitude = _moment_gaussian(offsets, values, peak)
    else:
        sigma = max(float(np.sqrt(-1.0 / (2.0 * quadratic))), 0.5)
        centre = float(-linear / (2.0 * quadratic))
        amplitude = float(np.exp(constant - linear**2 / (4.0 * quadratic)))
    coeffs = np.array([amplitude, centre, sigma], dtype=float)
    model = np.exp(-0.5 * ((offsets - centre) / sigma) ** 2)
    rms = float(np.sqrt(np.mean((amplitude * model - values) ** 2)))
    return ProfileFit("gaussian", coeffs, sigma, sigma * _FWHM_PER_SIGMA, rms)


def _degenerate_fit(offsets: np.ndarray, values: np.ndarray) -> ProfileFit:
    """Return a flat, unusable profile model for empty input."""
    coeffs = np.array([0.0, float(offsets.mean()) if offsets.size else 0.0, 1.0])
    return ProfileFit("gaussian", coeffs, 1.0, _FWHM_PER_SIGMA, float("nan"))


def _fit_polynomial(
    offsets: np.ndarray, values: np.ndarray, order: int
) -> ProfileFit:
    """Fit a Legendre polynomial to the profile in normalized offset space."""
    centre, sigma, _ = _moment_gaussian(offsets, values, float(values.max()))
    x = np.clip((offsets - centre) / (3.0 * sigma), -1.0, 1.0)
    legendre = np.polynomial.legendre.legfit(x, values, int(order))
    model = np.clip(np.polynomial.legendre.legval(x, legendre), 0.0, None)
    coeffs = np.concatenate([[centre, sigma], legendre])
    rms = float(np.sqrt(np.mean((model - values) ** 2)))
    half = 0.5 * float(values.max())
    above = np.where(model >= half)[0]
    fwhm = float(above.size) if above.size else sigma * _FWHM_PER_SIGMA
    return ProfileFit("polynomial", coeffs, sigma, max(fwhm, 1.0), rms)


def fit_object_profile(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    model: str = "gaussian",
    order: int = 2,
) -> ProfileFit:
    """Fit a spatial object profile to collapsed samples.

    Args:
        offsets: Pixel offset of each sample from the object trace.
        values: Collapsed object profile values.
        model: ``"gaussian"`` or ``"polynomial"`` profile parameterization.
        order: Legendre polynomial order when ``model == "polynomial"``.

    Returns:
        The fitted :class:`ProfileFit`.
    """
    offsets = np.asarray(offsets, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(offsets) & np.isfinite(values)
    if good.sum() < 3:
        return _degenerate_fit(offsets[good], values[good])
    offsets = offsets[good]
    values = values[good]
    if model == "polynomial":
        return _fit_polynomial(offsets, values, order)
    if model != "gaussian":
        raise ValueError(
            f"unknown profile model {model!r}; expected 'gaussian' or 'polynomial'"
        )
    return _fit_gaussian(offsets, values)


def collapse_object_profile(
    image: np.ndarray,
    ivar: np.ndarray,
    trace: np.ndarray,
    *,
    half_width: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an order image into an inverse-variance weighted object profile.

    Each dispersion row contributes the pixels within ``half_width`` of its
    object trace, weighted by that pixel's inverse variance, producing a mean
    spatial profile as a function of integer pixel offset from the trace.

    Args:
        image: Sky-subtracted order image, shape ``(nspec, nspat)``.
        ivar: Inverse-variance image, same shape as ``image``.
        trace: Object spatial trace, shape ``(nspec,)``.
        half_width: Half-width of the collapse window, in pixels.
        mask: Optional good-pixel mask, same shape as ``image``.

    Returns:
        ``(offsets, profile)`` with the integer pixel offsets and the collapsed
        profile values (NaN where no pixel contributed).
    """
    image = np.asarray(image, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    trace = np.asarray(trace, dtype=float)
    nspec, nspat = image.shape
    half_width = int(max(1, round(float(half_width))))
    offsets = np.arange(-half_width, half_width + 1, dtype=int)
    good = np.isfinite(image) & (ivar > 0)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)

    numerator = np.zeros(offsets.size, dtype=float)
    denominator = np.zeros(offsets.size, dtype=float)
    for row in range(nspec):
        centre = trace[row]
        if not np.isfinite(centre):
            continue
        columns = int(round(centre)) + offsets
        inside = (columns >= 0) & (columns < nspat)
        if not inside.any():
            continue
        columns = columns[inside]
        row_good = good[row, columns]
        weights = np.where(row_good, ivar[row, columns], 0.0)
        numerator[inside] += weights * np.where(row_good, image[row, columns], 0.0)
        denominator[inside] += weights

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = numerator / denominator
    profile = np.where(denominator > 0, profile, np.nan)
    return offsets, profile


def boxcar_extract(
    image: np.ndarray,
    ivar: np.ndarray,
    trace: np.ndarray,
    box_radius: float,
    *,
    mask: np.ndarray | None = None,
) -> BoxcarExtraction:
    """Sum the flux in a box around the object trace.

    Args:
        image: Sky-subtracted order image, shape ``(nspec, nspat)``.
        ivar: Inverse-variance image, same shape as ``image``.
        trace: Object spatial trace, shape ``(nspec,)``.
        box_radius: Half-width of the extraction box, in pixels.
        mask: Optional good-pixel mask, same shape as ``image``.

    Returns:
        The :class:`BoxcarExtraction`, including the propagated variance.
    """
    image = np.asarray(image, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    trace = np.asarray(trace, dtype=float)
    nspec, nspat = image.shape
    radius = float(box_radius)

    flux = np.full(nspec, np.nan)
    variance = np.full(nspec, np.nan)
    npix = np.zeros(nspec, dtype=float)
    good_spectrum = np.zeros(nspec, dtype=bool)

    for row in range(nspec):
        centre = trace[row]
        if not np.isfinite(centre):
            continue
        low = int(np.ceil(centre - radius))
        high = int(np.floor(centre + radius))
        if high < low:
            continue
        columns = np.arange(low, high + 1)
        inside = (columns >= 0) & (columns < nspat)
        if not inside.any():
            continue
        columns = columns[inside]
        good = np.isfinite(image[row, columns]) & (ivar[row, columns] > 0)
        if mask is not None:
            good &= np.asarray(mask, dtype=bool)[row, columns]
        if not good.any():
            continue
        flux[row] = float(image[row, columns][good].sum())
        variance[row] = float((1.0 / ivar[row, columns][good]).sum())
        npix[row] = float(good.sum())
        good_spectrum[row] = True

    with np.errstate(invalid="ignore", divide="ignore"):
        ivar_out = np.where(good_spectrum, 1.0 / variance, 0.0)
    return BoxcarExtraction(
        flux=flux,
        var=variance,
        ivar=ivar_out,
        npix=npix,
        mask=good_spectrum,
    )


def build_profile_image(
    fit: ProfileFit,
    trace: np.ndarray,
    shape: tuple[int, int],
    *,
    half_width: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a fitted profile as a per-row normalized 2D image.

    Args:
        fit: The fitted object profile.
        trace: Object spatial trace, shape ``(nspec,)``.
        shape: Output image shape ``(nspec, nspat)``.
        half_width: Half-width of the profile support, in pixels.
        mask: Optional good-pixel mask; masked pixels are zeroed after
            normalization so they carry no profile weight.

    Returns:
        A ``(nspec, nspat)`` profile image whose rows sum to one over the
        unmasked support.
    """
    nspec, nspat = shape
    trace = np.asarray(trace, dtype=float)[:, None]
    columns = np.arange(nspat, dtype=float)[None, :]
    offsets = columns - trace
    profile = fit.evaluate(offsets)
    profile = np.where(np.abs(offsets) <= float(half_width), profile, 0.0)
    profile = np.where(np.isfinite(profile) & np.isfinite(trace), profile, 0.0)
    norm = profile.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.divide(profile, norm, out=np.zeros_like(profile), where=norm > 0)
    if mask is not None:
        profile = np.where(np.asarray(mask, dtype=bool), profile, 0.0)
    return profile


def _optimal_step(
    image: np.ndarray,
    ivar: np.ndarray,
    profile: np.ndarray,
    good: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform one Horne optimal-extraction pass; see :func:`optimal_extract`."""
    weighted = np.where(good, profile, 0.0)
    denominator = weighted.sum(axis=1)
    numerator2 = np.where(good, ivar * profile**2, 0.0).sum(axis=1)
    numerator = np.where(good, ivar * profile * image, 0.0).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        flux = numerator / numerator2
        ivar_out = numerator2 / denominator
    mask = (numerator2 > 0) & (denominator > 0) & np.isfinite(flux)
    flux = np.where(mask, flux, np.nan)
    frac_use = np.where(numerator2 > 0, denominator, 0.0)

    model = np.where(mask[:, None], flux[:, None] * profile, 0.0)
    residual2 = np.where(good, (image - model) ** 2 * ivar, 0.0)
    counts = good.sum(axis=1)
    chi2 = np.where(counts > 1, residual2.sum(axis=1) / np.maximum(counts - 1, 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where((ivar_out > 0) & mask, 1.0 / ivar_out, np.nan)
    return flux, variance, ivar_out, mask, frac_use, chi2


def reject_cosmic_rays(
    image: np.ndarray,
    ivar: np.ndarray,
    profile: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    *,
    sigclip: float = 5.0,
) -> np.ndarray:
    """Flag pixels that are positive outliers from the object profile model.

    Cosmic rays are positive spikes relative to ``flux * profile``. A pixel is
    rejected when its residual exceeds ``sigclip`` times the larger of its
    formal noise and the robust scatter of the residual ensemble, so a
    slightly mismatched profile does not cause wholesale rejection.

    Args:
        image: Sky-subtracted order image, shape ``(nspec, nspat)``.
        ivar: Inverse-variance image, same shape as ``image``.
        profile: Normalized per-row profile image, same shape as ``image``.
        flux: Current per-row flux estimate, shape ``(nspec,)``.
        good: Boolean mask of pixels still in use.
        sigclip: Rejection threshold in sigma.

    Returns:
        A boolean image marking pixels to reject as cosmic rays.
    """
    image = np.asarray(image, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    profile = np.asarray(profile, dtype=float)
    flux = np.asarray(flux, dtype=float)
    good = np.asarray(good, dtype=bool)

    model = np.where(np.isfinite(flux)[:, None], flux[:, None] * profile, 0.0)
    residual = image - model
    valid = good & (profile > 0) & np.isfinite(residual)
    if valid.any():
        median = float(np.median(residual[valid]))
        scatter = 1.4826 * float(np.median(np.abs(residual[valid] - median)))
    else:
        scatter = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        noise = np.where(ivar > 0, 1.0 / np.sqrt(np.fmax(ivar, 1e-12)), 0.0)
    threshold = float(sigclip) * np.maximum(noise, max(scatter, 1e-12))
    return valid & (residual > threshold) & np.isfinite(threshold)


def optimal_extract(
    image: np.ndarray,
    ivar: np.ndarray,
    profile: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    min_frac_use: float = 0.5,
    sigclip: float | None = 5.0,
    niter: int = 3,
) -> OptimalExtraction:
    """Perform Horne (1986) optimal extraction of a single object.

    The flux estimate is ``sum(ivar * P * D) / sum(ivar * P**2)`` and its
    inverse variance is ``sum(ivar * P**2) / sum(P)``, where ``P`` is the
    per-row normalized object profile. When ``sigclip`` is set, pixels whose
    residual from the model exceeds ``sigclip`` times their noise are rejected
    as cosmic rays and the extraction is repeated ``niter`` times.

    Args:
        image: Sky-subtracted order image, shape ``(nspec, nspat)``.
        ivar: Inverse-variance image, same shape as ``image``.
        profile: Normalized per-row profile image, same shape as ``image``.
        mask: Optional good-pixel mask, same shape as ``image``.
        min_frac_use: Minimum used profile fraction for a valid pixel.
        sigclip: Cosmic-ray rejection threshold in sigma, or ``None`` to skip.
        niter: Maximum number of cosmic-ray rejection iterations.

    Returns:
        The :class:`OptimalExtraction`.
    """
    image = np.asarray(image, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    profile = np.asarray(profile, dtype=float)
    good = np.isfinite(image) & np.isfinite(ivar) & (ivar > 0) & np.isfinite(profile)
    good &= profile > 0
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)

    cr_mask = np.zeros_like(good)
    flux, variance, ivar_out, spectrum_mask, frac_use, chi2 = _optimal_step(
        image, ivar, profile, good
    )

    if sigclip is not None:
        for _ in range(max(0, int(niter))):
            rejected = reject_cosmic_rays(
                image, ivar, profile, flux, good, sigclip=float(sigclip)
            )
            if not rejected.any():
                break
            good &= ~rejected
            cr_mask |= rejected
            flux, variance, ivar_out, spectrum_mask, frac_use, chi2 = _optimal_step(
                image, ivar, profile, good
            )

    spectrum_mask &= frac_use > float(min_frac_use)
    flux = np.where(spectrum_mask, flux, np.nan)
    variance = np.where(spectrum_mask, variance, np.nan)
    ivar_out = np.where(spectrum_mask, ivar_out, 0.0)
    return OptimalExtraction(
        flux=flux,
        var=variance,
        ivar=ivar_out,
        mask=spectrum_mask,
        frac_use=frac_use,
        chi2=chi2,
        cr_mask=cr_mask,
    )


__all__ = [
    "BoxcarExtraction",
    "OptimalExtraction",
    "ProfileFit",
    "boxcar_extract",
    "build_profile_image",
    "collapse_object_profile",
    "fit_object_profile",
    "optimal_extract",
    "reject_cosmic_rays",
]
