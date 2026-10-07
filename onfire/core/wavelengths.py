"""Pure Step Engines for 1D wavelength calibration.

The functions here detect arc/OH emission lines in an extracted order spectrum,
reidentify them against a bundled Magellan/FIRE wavelength archive, fit a 1D
Legendre dispersion solution per order, and fit a global 2D pixel-vs-order
solution across all orders. They operate only on plain NumPy arrays (ADR 0003)
and are used by the ``calibrate_wavelengths`` Step Engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onfire.core.polyfit import PolynomialFit, robust_legendre_fit

# The template archive is sampled on a fixed 2040-pixel grid per order.
ARXIV_NPIX: int = 2040


@dataclass
class DetectedLines:
    """Emission lines detected in an order arc spectrum.

    Attributes:
        tcent: Pixel centroids of the detected lines.
        tampl: Continuum-subtracted peak amplitudes.
        twid: Gaussian-equivalent line widths (pixels).
        nsig: Peak amplitudes in units of the local noise.
        cont_sub: The continuum-subtracted spectrum.
        good: Boolean mask marking detections above the significance threshold.
    """

    tcent: np.ndarray
    tampl: np.ndarray
    twid: np.ndarray
    nsig: np.ndarray
    cont_sub: np.ndarray
    good: np.ndarray


@dataclass
class WavelengthSolution:
    """A 1D Legendre dispersion solution for one echelle order.

    Attributes:
        coeff: Legendre coefficients approximating the 1D solution;
            :attr:`wave_soln` is the authoritative evaluation.
        xmid: Dispersion coordinate midpoint used for normalization.
        xscale: Dispersion coordinate scale used for normalization.
        rms: RMS of the fit, in pixels.
        nlines: Number of identified lines used in the final fit.
        pixel_fit: Pixel positions of the identified lines.
        wave_fit: Wavelengths assigned to ``pixel_fit``.
        wave_soln: The solution evaluated at every detector pixel.
        cen_wave: Wavelength at the order centre.
        cen_disp: Wavelength dispersion per pixel at the order centre.
        masked: Whether the fit failed and the order should be masked.
    """

    coeff: np.ndarray
    xmid: float
    xscale: float
    rms: float
    nlines: int
    pixel_fit: np.ndarray
    wave_fit: np.ndarray
    wave_soln: np.ndarray
    cen_wave: float
    cen_disp: float
    masked: bool = False


def trace_xcoord(nspec: int) -> tuple[float, float]:
    """Return the Legendre dispersion-coordinate midpoint and scale."""
    xmid = (nspec - 1) / 2.0
    return xmid, (xmid if xmid > 0 else 1.0)


def _median_filter(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size <= 1:
        return values.copy()
    half = size // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


def iter_continuum(
    spec: np.ndarray,
    *,
    fwhm: float = 4.0,
    cont_samp: float = 30.0,
    niter_cont: int = 3,
    sigrej: float = 3.0,
) -> np.ndarray:
    """Estimate the continuum of an arc spectrum.

    A low-percentile running filter follows the continuum beneath the emission
    lines; it is then smoothed and iteratively sigma-clipped.

    Args:
        spec: The 1D arc spectrum.
        fwhm: Expected arc-line FWHM in pixels.
        cont_samp: Continuum sampling scale, in units of ``fwhm``.
        niter_cont: Number of rejection iterations.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        The estimated continuum, the same shape as ``spec``.
    """
    spec = np.asarray(spec, dtype=float)
    window = max(3, int(round(cont_samp * fwhm)))
    cont = _median_filter(spec, window)
    for _ in range(max(0, niter_cont)):
        resid = spec - cont
        scatter = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scatter <= 0:
            break
        clip = np.minimum(spec, cont + sigrej * scatter)
        cont = _median_filter(clip, window)
    return cont


def _local_maxima(spec: np.ndarray) -> np.ndarray:
    interior = (spec[1:-1] > spec[:-2]) & (spec[1:-1] >= spec[2:])
    return np.where(interior)[0] + 1


def _select_peaks(
    spec: np.ndarray, candidates: np.ndarray, *, min_sep: float
) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    order = candidates[np.argsort(-spec[candidates])]
    kept: list[int] = []
    for peak in order:
        if all(abs(peak - chosen) >= min_sep for chosen in kept):
            kept.append(int(peak))
    return np.sort(np.asarray(kept, dtype=int))


def detect_lines(
    spec: np.ndarray,
    *,
    sigdetect: float = 5.0,
    fwhm: float = 4.0,
    cont_frac_fwhm: float = 1.0,
    fit_frac_fwhm: float = 1.25,
    cont_samp: float = 30.0,
    niter_cont: int = 3,
) -> DetectedLines:
    """Detect emission lines in a 1D arc spectrum.

    Args:
        spec: The 1D arc spectrum.
        sigdetect: Detection threshold in units of the local noise.
        fwhm: Expected line FWHM in pixels, used for centroid windows and the
            minimum line separation.
        cont_frac_fwhm: Continuum sampling scale.
        fit_frac_fwhm: Centroid half-window, in units of ``fwhm``.
        cont_samp: Continuum sampling scale passed to :func:`iter_continuum`.
        niter_cont: Continuum rejection iterations.

    Returns:
        A :class:`DetectedLines` describing the detections.
    """
    spec = np.asarray(spec, dtype=float)
    cont = iter_continuum(
        spec, fwhm=fwhm, cont_samp=cont_samp * cont_frac_fwhm, niter_cont=niter_cont
    )
    cont_sub = spec - cont
    scatter = 1.4826 * np.median(np.abs(cont_sub - np.median(cont_sub)))
    if scatter <= 0:
        scatter = float(np.std(cont_sub)) or 1.0

    threshold = sigdetect * scatter
    candidates = _local_maxima(cont_sub)
    candidates = candidates[cont_sub[candidates] > threshold]
    peaks = _select_peaks(cont_sub, candidates, min_sep=max(1.0, fwhm))

    half = max(1, int(round(fit_frac_fwhm * fwhm)))
    nspec = cont_sub.size
    tcent = np.empty(peaks.size)
    tampl = np.empty(peaks.size)
    twid = np.empty(peaks.size)
    for i, peak in enumerate(peaks):
        lo = max(0, peak - half)
        hi = min(nspec, peak + half + 1)
        segment = np.clip(cont_sub[lo:hi], 0.0, None)
        total = segment.sum()
        columns = np.arange(lo, hi, dtype=float)
        if total > 0:
            centroid = float((columns * segment).sum() / total)
            width = float(np.sqrt((segment * (columns - centroid) ** 2).sum() / total))
        else:
            centroid = float(peak)
            width = float(fwhm)
        tcent[i] = centroid
        twid[i] = max(width, 1.0)
        tampl[i] = float(np.interp(centroid, np.arange(nspec), cont_sub))

    nsig = tampl / scatter
    good = nsig >= sigdetect
    return DetectedLines(
        tcent=tcent[good],
        tampl=tampl[good],
        twid=twid[good],
        nsig=nsig[good],
        cont_sub=cont_sub,
        good=good,
    )


def _prep_xcorr(values: np.ndarray, percent_ceil: float) -> np.ndarray:
    values = np.asarray(values, dtype=float) - np.median(values)
    if percent_ceil is not None:
        values = np.minimum(values, np.percentile(values, percent_ceil))
    norm = np.sqrt(np.sum(values**2))
    return values / norm if norm > 0 else values


def xcorr_shift(
    spec: np.ndarray,
    template: np.ndarray,
    *,
    max_lag_frac: float = 1.0,
    percent_ceil: float = 50.0,
) -> tuple[float, float]:
    """Cross-correlate an observed spectrum against a template.

    Args:
        spec: Observed arc spectrum.
        template: Template arc spectrum on the same pixel grid.
        max_lag_frac: Maximum lag to search, as a fraction of the spectrum
            length.
        percent_ceil: Upper percentile used to suppress the strongest lines
            before correlating.

    Returns:
        ``(shift, corr)`` where ``shift`` is the pixel shift such that
        ``template_pixel = observed_pixel - shift`` and ``corr`` is the
        normalized peak correlation.
    """
    spec = np.asarray(spec, dtype=float)
    template = np.asarray(template, dtype=float)
    n = spec.size
    if template.size != n:
        raise ValueError("spec and template must have the same length")
    a = _prep_xcorr(spec, percent_ceil)
    b = _prep_xcorr(template, percent_ceil)
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)
    lags = np.arange(n)
    lags = np.where(lags > n // 2, lags - n, lags)
    max_lag = int(max_lag_frac * n)
    allowed = np.abs(lags) <= max_lag
    masked = np.where(allowed, corr, -np.inf)
    best = int(np.argmax(masked))
    peak_lag = int(lags[best])
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    peak_corr = float(corr[best] / denom) if denom > 0 else 0.0
    return float(peak_lag), peak_corr


def _resample(values: np.ndarray, nspec: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == nspec:
        return values.copy()
    return np.interp(np.linspace(0.0, values.size - 1, nspec), np.arange(values.size), values)


def fit_wavelength_solution(
    pixel: np.ndarray,
    wave: np.ndarray,
    *,
    ncoeff: int,
    nspec: int | None = None,
    sigrej: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a 1D Legendre pixel-to-wavelength solution.

    Args:
        pixel: Pixel positions of the identified lines.
        wave: Assigned wavelengths.
        ncoeff: Number of Legendre coefficients.
        nspec: Detector length, fixing the normalization used for evaluation.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        ``(coeff, good, rms)`` with the coefficient array, the retained-point
        mask, and the RMS residual in pixels.
    """
    pixel = np.asarray(pixel, dtype=float)
    wave = np.asarray(wave, dtype=float)
    if nspec is None:
        nspec = int(np.ceil(pixel.max())) + 1 if pixel.size else 1
    fit, good, _ = robust_legendre_fit(
        [pixel],
        wave,
        (max(0, ncoeff - 1),),
        domains=[(0.0, float(max(1, nspec - 1)))],
        sigrej=sigrej,
    )
    dispersion = _pixel_dispersion(pixel, wave)
    rms_pixels = float("nan")
    if np.any(good) and np.isfinite(dispersion) and dispersion != 0:
        residual = wave[good] - fit.evaluate([pixel[good]])
        rms_pixels = float(np.sqrt(np.mean((residual / dispersion) ** 2)))
    return np.asarray(fit.coeffs, dtype=float), good, rms_pixels


def _match_to_line_list(
    pixels: np.ndarray,
    guesses: np.ndarray,
    line_waves: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predicted wavelengths to the nearest rest wavelengths."""
    pixel_fit: list[float] = []
    wave_fit: list[float] = []
    for pixel, guess in zip(
        np.asarray(pixels, dtype=float), np.asarray(guesses, dtype=float)
    ):
        index = int(np.argmin(np.abs(line_waves - guess)))
        if abs(line_waves[index] - guess) < tolerance:
            pixel_fit.append(float(pixel))
            wave_fit.append(float(line_waves[index]))
    return np.asarray(pixel_fit, dtype=float), np.asarray(wave_fit, dtype=float)


def _pixel_dispersion(pixel: np.ndarray, wave: np.ndarray) -> float:
    pixel = np.asarray(pixel, dtype=float)
    wave = np.asarray(wave, dtype=float)
    order = np.argsort(pixel)
    if pixel.size < 2:
        return float("nan")
    return float(abs(np.median(np.diff(wave[order]) / np.diff(pixel[order]))))


def evaluate_wavelength_solution(
    coeff: np.ndarray,
    xmid: float,
    xscale: float,
    rows: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted 1D Legendre dispersion solution at ``rows``."""
    x = (np.asarray(rows, dtype=float) - xmid) / xscale
    return np.polynomial.legendre.legval(x, np.asarray(coeff, dtype=float))


def _solution_from_pixels(
    pixel: np.ndarray,
    wave: np.ndarray,
    nspec: int,
    *,
    ncoeff: int,
    sigrej: float,
) -> WavelengthSolution:
    coeff, good, rms = fit_wavelength_solution(
        pixel, wave, ncoeff=ncoeff, nspec=nspec, sigrej=sigrej
    )
    xmid, xscale = trace_xcoord(nspec)
    rows = np.arange(nspec)
    wave_soln = evaluate_wavelength_solution(coeff, xmid, xscale, rows)
    cen_wave = float(evaluate_wavelength_solution(coeff, xmid, xscale, np.array([nspec // 2]))[0])
    cen_disp = (
        float(wave_soln[min(nspec - 1, nspec // 2 + 1)] - wave_soln[nspec // 2])
        if nspec > 1
        else 0.0
    )
    return WavelengthSolution(
        coeff=coeff,
        xmid=xmid,
        xscale=xscale,
        rms=rms,
        nlines=int(good.sum()),
        pixel_fit=pixel[good],
        wave_fit=wave[good],
        wave_soln=wave_soln,
        cen_wave=cen_wave,
        cen_disp=cen_disp,
        masked=False,
    )


def reidentify_spectrum(
    spec: np.ndarray,
    arxiv_flux: np.ndarray,
    arxiv_wave: np.ndarray,
    line_waves: np.ndarray,
    *,
    sigdetect: float = 5.0,
    fwhm: float = 4.0,
    match_toler: float = 2.0,
    n_first: int = 2,
    n_final: int = 4,
    sigrej: float = 3.0,
    niter: int = 6,
) -> WavelengthSolution:
    """Determine a 1D wavelength solution by reidentifying the archive.

    The observed spectrum is cross-correlated with the bundled archive arc
    spectrum to obtain a pixel shift, each detected line is matched to the
    closest rest wavelength in the line list, and a Legendre solution is fit
    iteratively while new line matches are added.

    Args:
        spec: Extracted arc spectrum for one order.
        arxiv_flux: Archived composite arc spectrum (2040 pixels).
        arxiv_wave: Archived wavelength solution on the archive pixel grid.
        line_waves: Rest wavelengths of the OH line list.
        sigdetect: Line-detection threshold in noise sigma.
        fwhm: Expected line FWHM in pixels.
        match_toler: Wavelength-matching tolerance in pixels.
        n_first: Polynomial order of the first-pass fit.
        n_final: Polynomial order of the final fit.
        sigrej: Rejection threshold in robust sigma.
        niter: Maximum number of line-addition iterations.

    Returns:
        The fitted :class:`WavelengthSolution`.
    """
    spec = np.asarray(spec, dtype=float)
    nspec = spec.size
    line_waves = np.sort(np.asarray(line_waves, dtype=float))
    arxiv_flux_rs = _resample(arxiv_flux, nspec)
    arxiv_wave_rs = _resample(arxiv_wave, nspec)
    dispersion = abs(float(np.median(np.diff(arxiv_wave_rs))))
    tolerance = max(match_toler * dispersion, 1e-6)

    detected = detect_lines(spec, sigdetect=sigdetect, fwhm=fwhm)
    if detected.tcent.size < 3:
        return _masked_solution(nspec, ncoeff=n_first + 1)

    shift, _ = xcorr_shift(spec, arxiv_flux_rs)
    template_pixels = detected.tcent - shift
    wave_guess = np.interp(template_pixels, np.arange(nspec), arxiv_wave_rs)
    pixel_fit, wave_fit = _match_to_line_list(
        detected.tcent, wave_guess, line_waves, tolerance
    )

    if pixel_fit.size < 3:
        return _masked_solution(nspec, ncoeff=n_first + 1)

    solution = _solution_from_pixels(
        pixel_fit, wave_fit, nspec, ncoeff=n_first + 1, sigrej=sigrej
    )
    for _ in range(max(0, niter)):
        predicted = evaluate_wavelength_solution(
            solution.coeff, solution.xmid, solution.xscale, detected.tcent
        )
        extra_pixel, extra_wave = _match_to_line_list(
            detected.tcent, predicted, line_waves, tolerance
        )
        merged_pixel = np.concatenate([pixel_fit, extra_pixel])
        merged_wave = np.concatenate([wave_fit, extra_wave])
        unique, indices = np.unique(np.round(merged_pixel, 4), return_index=True)
        merged_pixel = unique
        merged_wave = merged_wave[indices]
        ncoeff = min(n_final + 1, max(n_first + 1, solution.coeff.size + 1))
        updated = _solution_from_pixels(
            merged_pixel, merged_wave, nspec, ncoeff=ncoeff, sigrej=sigrej
        )
        if updated.nlines <= solution.nlines and ncoeff == solution.coeff.size:
            solution = updated
            break
        solution = updated
    return solution


def _masked_solution(nspec: int, ncoeff: int = 4) -> WavelengthSolution:
    xmid, xscale = trace_xcoord(nspec)
    return WavelengthSolution(
        coeff=np.full(ncoeff, np.nan),
        xmid=xmid,
        xscale=xscale,
        rms=float("nan"),
        nlines=0,
        pixel_fit=np.array([]),
        wave_fit=np.array([]),
        wave_soln=np.full(nspec, np.nan),
        cen_wave=float("nan"),
        cen_disp=float("nan"),
        masked=True,
    )


def regularize_wavelength_solution(
    spec: np.ndarray,
    line_waves: np.ndarray,
    prior_wave: np.ndarray,
    *,
    sigdetect: float = 5.0,
    fwhm: float = 4.0,
    match_toler: float = 3.0,
    ncoeff: int = 5,
    correction_order: int = 1,
    sigrej: float = 3.0,
) -> WavelengthSolution:
    """Refine one order using cross-order regularization from a global prior.

    The global solution ``prior_wave`` provides predicted wavelengths at every
    pixel. Newly detected lines are matched to the line list and a smooth,
    low-order correction to the global solution is fit. Orders with too few
    matches fall back to the global prior, so faint orders inherit the
    instrument solution instead of failing.

    Args:
        spec: Extracted arc spectrum for the target order.
        line_waves: Rest wavelengths of the OH line list.
        prior_wave: Global wavelength solution evaluated at every pixel, shape
            ``(nspec,)``.
        sigdetect: Line-detection threshold in noise sigma.
        fwhm: Expected line FWHM in pixels.
        match_toler: Matching tolerance in pixels.
        ncoeff: Number of coefficients of the stored 1D solution.
        correction_order: Polynomial order of the smooth correction.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        The regularized :class:`WavelengthSolution`.
    """
    spec = np.asarray(spec, dtype=float)
    prior_wave = np.asarray(prior_wave, dtype=float)
    nspec = spec.size
    rows = np.arange(nspec, dtype=float)
    line_waves = np.sort(np.asarray(line_waves, dtype=float))
    dispersion = abs(float(np.median(np.diff(prior_wave))))
    tolerance = max(match_toler * dispersion, 1e-6)

    detected = detect_lines(spec, sigdetect=sigdetect, fwhm=fwhm)
    guess = np.interp(detected.tcent, rows, prior_wave)
    pixel_fit, wave_fit = _match_to_line_list(
        detected.tcent, guess, line_waves, tolerance
    )

    if pixel_fit.size < 3:
        return _fallback_solution(prior_wave, ncoeff=ncoeff)

    prior_at_lines = np.interp(pixel_fit, rows, prior_wave)
    residual = wave_fit - prior_at_lines
    fit, _, _ = robust_legendre_fit(
        [pixel_fit],
        residual,
        (max(0, correction_order),),
        domains=[(0.0, float(max(1, nspec - 1)))],
        sigrej=sigrej,
    )
    correction = fit.evaluate([rows])
    wave_soln = prior_wave + correction

    coeff, _, rms = fit_wavelength_solution(
        rows, wave_soln, ncoeff=ncoeff, nspec=nspec, sigrej=sigrej
    )
    xmid, xscale = trace_xcoord(nspec)
    return WavelengthSolution(
        coeff=coeff,
        xmid=xmid,
        xscale=xscale,
        rms=rms,
        nlines=int(pixel_fit.size),
        pixel_fit=pixel_fit,
        wave_fit=wave_fit,
        wave_soln=wave_soln,
        cen_wave=float(wave_soln[nspec // 2]),
        cen_disp=float(wave_soln[min(nspec - 1, nspec // 2 + 1)] - wave_soln[nspec // 2]),
        masked=False,
    )


def _fallback_solution(prior_wave: np.ndarray, *, ncoeff: int) -> WavelengthSolution:
    nspec = prior_wave.size
    rows = np.arange(nspec, dtype=float)
    coeff, _, rms = fit_wavelength_solution(
        rows, prior_wave, ncoeff=ncoeff, nspec=nspec
    )
    xmid, xscale = trace_xcoord(nspec)
    return WavelengthSolution(
        coeff=coeff,
        xmid=xmid,
        xscale=xscale,
        rms=rms,
        nlines=0,
        pixel_fit=np.array([]),
        wave_fit=np.array([]),
        wave_soln=np.asarray(prior_wave, dtype=float).copy(),
        cen_wave=float(prior_wave[nspec // 2]),
        cen_disp=float(prior_wave[min(nspec - 1, nspec // 2 + 1)] - prior_wave[nspec // 2]),
        masked=False,
    )


def fit_wavelength_surface(
    pixel: np.ndarray,
    wave: np.ndarray,
    order: np.ndarray,
    *,
    nspec: int,
    order_min: int,
    order_max: int,
    nspec_coeff: int = 4,
    norder_coeff: int = 6,
    sigrej: float = 3.0,
) -> tuple[PolynomialFit, float]:
    """Fit the global 2D pixel-vs-order wavelength surface.

    Args:
        pixel: Detector pixel positions of all identified lines.
        wave: Assigned wavelengths.
        order: Echelle order number of each line.
        nspec: Number of detector pixels along dispersion.
        order_min: Minimum echelle order number.
        order_max: Maximum echelle order number.
        nspec_coeff: Polynomial order along the pixel axis.
        norder_coeff: Polynomial order along the order axis.
        sigrej: Rejection threshold in robust sigma.

    Returns:
        ``(fit, rms)`` with the fitted surface (normalized pixel and order
        coordinates in ``[0, 1]``) and its RMS residual in Angstrom.
    """
    pixel = np.asarray(pixel, dtype=float)
    wave = np.asarray(wave, dtype=float)
    order = np.asarray(order, dtype=float)
    span = float(max(1, order_max - order_min))
    spec_coord = pixel / float(max(1, nspec - 1))
    order_coord = (order - order_min) / span
    fit, _, _ = robust_legendre_fit(
        [spec_coord, order_coord],
        wave,
        (max(0, nspec_coeff - 1), max(0, norder_coeff - 1)),
        domains=[(0.0, 1.0), (0.0, 1.0)],
        sigrej=sigrej,
    )
    residual = wave - fit.evaluate([spec_coord, order_coord])
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")
    return fit, rms


def evaluate_wavelength_surface(
    fit: PolynomialFit,
    pixel: np.ndarray,
    order: np.ndarray,
    *,
    nspec: int,
    order_min: int,
    order_max: int,
) -> np.ndarray:
    """Evaluate a global wavelength surface at (pixel, order) positions."""
    span = float(max(1, order_max - order_min))
    spec_coord = np.asarray(pixel, dtype=float) / float(max(1, nspec - 1))
    order_coord = (np.asarray(order, dtype=float) - order_min) / span
    return fit.evaluate([spec_coord, order_coord])


__all__ = [
    "DetectedLines",
    "WavelengthSolution",
    "detect_lines",
    "evaluate_wavelength_solution",
    "evaluate_wavelength_surface",
    "fit_wavelength_solution",
    "fit_wavelength_surface",
    "iter_continuum",
    "reidentify_spectrum",
    "regularize_wavelength_solution",
    "trace_xcoord",
    "xcorr_shift",
]
