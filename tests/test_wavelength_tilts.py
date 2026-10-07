"""High-level tests for wavelength calibration and spectral line tilts.

These exercise the public seams for reduction steps 3 and 4: the bundled FIRE
reference data, the pure ``onfire.core`` line-detection / reidentification /
tilt-fitting functions, and the ``ReductionState`` API that persists the
wavelength and tilt products to Multi-Extension FITS with Plotly QA artifacts.
"""

import numpy as np
import pytest
from astropy.io import fits

from onfire.core import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
    build_tilt_fit,
    calibrate_wavelengths,
    detect_lines,
    evaluate_tilt_surface,
    evaluate_wavelength_solution,
    fit_tilt_surface,
    fit_wavelength_surface,
    fit_tilts,
    register_engines,
    regularize_wavelength_solution,
    reidentify_spectrum,
    tilt_angle,
    trace_tilts,
    xcorr_shift,
)
from onfire.data import load_fire_line_list, load_fire_template
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable, scan_raw_dir


@pytest.fixture(autouse=True)
def _engines():
    clear_step_engines()
    register_engines()
    yield
    clear_step_engines()


def make_flat(nspec=300, nspat=600, *, sigma=6.0, amplitude=200.0, background=20.0, seed=0):
    """Build a synthetic FIRE-like flat of separated Gaussian order stripes."""
    rng = np.random.default_rng(seed)
    centers = FIRE_ORDER_SPAT_POS * (nspat - 1)
    cols = np.arange(nspat, dtype=float)[None, :]
    flat = np.full((nspec, nspat), background, dtype=float)
    for center in centers:
        flat += amplitude * np.exp(-0.5 * ((cols - center) / sigma) ** 2)
    flat += rng.normal(0.0, 0.5, size=flat.shape)
    return flat


def make_arc_image(nspec=300, nspat=600, *, sigma=6.0, tilt=0.03, seed=1, background=5.0):
    """Build a synthetic arc image from the bundled FIRE archive.

    Each order's archived composite arc spectrum is resampled onto the
    detector and stamped across the order spatial profile, with the spectral
    position shifted linearly with spatial offset to imitate a line tilt.
    """
    rng = np.random.default_rng(seed)
    template = load_fire_template()
    centers = FIRE_ORDER_SPAT_POS * (nspat - 1)
    rows = np.arange(nspec, dtype=float)[:, None]
    cols = np.arange(nspat, dtype=float)[None, :]
    image = np.full((nspec, nspat), background, dtype=float)
    for i, order in enumerate(np.asarray(template["order"], dtype=int)):
        if order not in FIRE_ORDER_NUMBERS:
            continue
        flux = np.abs(np.asarray(template["flux"][i], dtype=float))
        spectrum = np.interp(np.linspace(0, flux.size - 1, nspec), np.arange(flux.size), flux)
        spectrum = spectrum / max(np.median(spectrum[spectrum > 0]), 1e-6)
        center = centers[int(np.where(FIRE_ORDER_NUMBERS == order)[0][0])]
        profile = np.exp(-0.5 * ((cols - center) / sigma) ** 2)
        shifted = np.interp(
            rows - tilt * (cols - center), np.arange(nspec), spectrum, left=0.0, right=0.0
        )
        image += shifted * profile
    image += rng.normal(0.0, 0.02 * np.median(image), image.shape)
    return image


def _order_solution(spec, order, **kwargs):
    template = load_fire_template()
    index = int(np.where(np.asarray(template["order"], dtype=int) == order)[0][0])
    arxiv_wave = np.asarray(template["wave"][index], dtype=float)
    arxiv_flux = np.abs(np.asarray(template["flux"][index], dtype=float))
    line_waves = np.asarray(load_fire_line_list()["wave"], dtype=float)
    return reidentify_spectrum(spec, arxiv_flux, arxiv_wave, line_waves, **kwargs)


def test_bundled_reference_data_loads():
    template = load_fire_template()
    line_list = load_fire_line_list()

    assert len(template) == FIRE_NORDERS
    assert list(np.asarray(template["order"], dtype=int)) == list(FIRE_ORDER_NUMBERS)
    assert len(template["wave"][0]) == 2040
    assert len(template["flux"][0]) == 2040
    assert len(line_list) > 100
    assert {"wave", "ion", "amplitude"} <= set(line_list.colnames)
    assert np.all(np.diff(np.asarray(line_list["wave"], dtype=float)) >= 0)


def test_detect_lines_recovers_gaussian_peaks():
    nspec = 400
    pixels = np.array([60.0, 150.5, 241.0, 330.25])
    amps = np.array([10.0, 20.0, 5.0, 15.0])
    spec = np.full(nspec, 2.0)
    grid = np.arange(nspec, dtype=float)
    for pixel, amp in zip(pixels, amps):
        spec += amp * np.exp(-0.5 * ((grid - pixel) / 2.0) ** 2)
    spec += np.random.default_rng(11).normal(0.0, 0.1, nspec)

    detected = detect_lines(spec, sigdetect=5.0, fwhm=4.0)

    assert detected.tcent.size == pixels.size
    np.testing.assert_allclose(np.sort(detected.tcent), pixels, atol=0.2)
    assert np.all(detected.nsig > 5.0)


def test_xcorr_shift_recovers_offset():
    rng = np.random.default_rng(0)
    template = rng.normal(size=500)
    template[[100, 240, 380]] += [10.0, 8.0, 6.0]

    for shift in (11, -7, 0):
        observed = np.roll(template, shift)
        recovered, corr = xcorr_shift(observed, template)
        assert recovered == pytest.approx(shift, abs=1.0)
        assert corr > 0.5


def test_reidentify_recovers_template_solution():
    template = load_fire_template()
    index = int(np.where(np.asarray(template["order"], dtype=int) == 21)[0][0])
    flux = np.abs(np.asarray(template["flux"][index], dtype=float))
    nspec = 2040
    rng = np.random.default_rng(2)
    spec = np.interp(np.linspace(0, flux.size - 1, nspec), np.arange(flux.size), flux)
    spec = spec / np.median(spec[spec > 0]) + rng.normal(0, 0.02, nspec)

    solution = _order_solution(spec, 21)

    assert not solution.masked
    assert solution.nlines >= 8
    assert solution.rms < 0.5
    truth = np.asarray(template["wave"][index], dtype=float)
    error = np.abs(solution.wave_soln - truth)
    assert np.median(error) < 1.0


def test_fit_wavelength_surface_recovers_known_surface():
    rng = np.random.default_rng(3)
    nspec = 500
    pixel = rng.uniform(0, nspec - 1, 400)
    order = rng.integers(11, 32, 400).astype(float)
    wave = 8000.0 + 6.0 * pixel + 40.0 * (order - 21) + 0.002 * pixel * (order - 21)

    fit, rms = fit_wavelength_surface(
        pixel, wave, order, nspec=nspec, order_min=11, order_max=31,
        nspec_coeff=3, norder_coeff=4,
    )

    assert rms < 1e-6
    check_pixel = np.array([100.0, 250.0, 400.0])
    check_order = np.array([13.0, 21.0, 29.0])
    expected = 8000.0 + 6.0 * check_pixel + 40.0 * (check_order - 21) + 0.002 * check_pixel * (check_order - 21)
    predicted = fit.evaluate(
        [
            check_pixel / (nspec - 1),
            (check_order - 11) / 20.0,
        ]
    )
    np.testing.assert_allclose(predicted, expected, atol=1e-6)


def test_regularize_uses_global_prior_for_faint_order():
    nspec = 500
    rows = np.arange(nspec, dtype=float)
    prior = 9000.0 + 5.0 * rows
    line_waves = np.sort(np.asarray(load_fire_line_list()["wave"], dtype=float))
    selected = line_waves[(line_waves > prior[0] + 20) & (line_waves < prior[-1] - 20)][::25][:8]
    line_pixels = (selected - 9000.0) / 5.0
    spec = np.full(nspec, 1.0)
    for pixel in line_pixels:
        spec += 10.0 * np.exp(-0.5 * ((rows - pixel) / 2.0) ** 2)
    spec += np.random.default_rng(4).normal(0, 0.02, nspec)

    solution = regularize_wavelength_solution(
        spec, line_waves, prior, sigdetect=5.0, fwhm=4.0, ncoeff=4
    )

    assert not solution.masked
    assert solution.nlines >= 4
    residual = solution.wave_soln - prior
    assert np.max(np.abs(residual)) < 1.0


def _traced(flat, arc):
    from onfire.core import trace_orders

    return trace_orders({"flat_norm": flat}, {}, {"ncoeff": 4}).products["order_traces"]


def test_calibrate_wavelengths_engine_full_run():
    flat = make_flat()
    arc = make_arc_image()
    traces = _traced(flat, arc)

    result = calibrate_wavelengths(
        {"order_traces": traces, "arc_image": arc}, {}, {"ncoeff": 5}
    )

    table = result.products["wavelengths"]
    assert len(table) == FIRE_NORDERS
    assert result.meta["n_orders_calibrated"] >= 15
    assert result.products["wave_soln"].shape == (arc.shape[0], FIRE_NORDERS)
    assert result.products["wave_surface"].shape == (arc.shape[0], FIRE_NORDERS)
    unmasked = table[~np.asarray(table["masked"], dtype=bool)]
    assert np.all(np.isfinite(np.asarray(unmasked["cen_wave"], dtype=float)))
    assert "wavelength_residuals" in result.plots
    assert "wavelength_identifications" in result.plots


def test_calibrate_wavelengths_single_order_preserves_others():
    flat = make_flat()
    arc = make_arc_image()
    traces = _traced(flat, arc)
    full = calibrate_wavelengths(
        {"order_traces": traces, "arc_image": arc}, {}, {"ncoeff": 5}
    )
    before = full.products["wave_soln"].copy()
    table = full.products["wavelengths"]
    unmasked = FIRE_ORDER_NUMBERS[~np.asarray(table["masked"], dtype=bool)]
    target = int(unmasked[len(unmasked) // 2])

    products = {"order_traces": traces, "arc_image": arc, **full.products}
    refined = calibrate_wavelengths(products, {}, {"ncoeff": 5, "order": target})

    after = refined.products["wave_soln"]
    index = int(np.where(FIRE_ORDER_NUMBERS == target)[0][0])
    assert np.all(np.isfinite(after[:, index]))
    for other in range(FIRE_NORDERS):
        if other == index:
            continue
        np.testing.assert_array_equal(after[:, other], before[:, other])


def test_fit_tilts_angle_matches_injected_slope():
    nspec, nspat = 300, 90
    center_spat = 45.0
    half_width = 30.0
    slope = 0.2
    line_specs = np.array([40.0, 90.0, 140.0, 190.0, 240.0])
    rows = np.arange(nspec, dtype=float)[:, None]
    cols = np.arange(nspat, dtype=float)[None, :]
    image = np.zeros((nspec, nspat))
    for line in line_specs:
        position = line + slope * (cols - center_spat)
        image += np.exp(-0.5 * ((rows - position) / 2.0) ** 2)

    measurements = trace_tilts(
        image, line_specs, center_spat, half_width=half_width, fwhm=4.0
    )
    fit = fit_tilt_surface(
        measurements, nspec=nspec, spat_scale=half_width, spec_order=3, spat_order=4
    )

    assert not fit.masked
    assert fit.tilt_angle == pytest.approx(np.arctan(slope), abs=0.02)
    poly = build_tilt_fit(fit)
    at_center = evaluate_tilt_surface(
        poly, np.array([120.0]), np.array([0.0]), nspec=nspec, spat_scale=half_width
    )
    assert abs(float(at_center[0])) < 1.0


def test_fit_tilts_engine_and_single_order_regularization():
    flat = make_flat()
    arc = make_arc_image()
    traces = _traced(flat, arc)
    wave_result = calibrate_wavelengths(
        {"order_traces": traces, "arc_image": arc}, {}, {"ncoeff": 5}
    )
    products = {"order_traces": traces, **wave_result.products}

    full = fit_tilts(products, {}, {"tilt_sigdetect": 8.0})
    table = full.products["tilts"]
    assert len(table) == FIRE_NORDERS
    assert full.meta["n_orders_tilted"] >= 15
    assert "tilt_residuals" in full.plots
    assert "tilt_map" in full.plots

    before = full.products["tilt_coeffs"].copy()
    unmasked = FIRE_ORDER_NUMBERS[~np.asarray(table["masked"], dtype=bool)]
    target = int(unmasked[len(unmasked) // 2])
    refined = fit_tilts(
        {**products, **full.products}, {}, {"tilt_sigdetect": 8.0, "order": target}
    )
    after = refined.products["tilt_coeffs"]
    index = int(np.where(FIRE_ORDER_NUMBERS == target)[0][0])
    assert np.all(np.isfinite(after[index]))
    for other in range(FIRE_NORDERS):
        if other == index:
            continue
        np.testing.assert_array_equal(after[other], before[other])


def _write_frame(path, data, obstype, object_name):
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.header["OBSTYPE"] = obstype
    hdu.header["OBJECT"] = object_name
    hdu.header["EXPTIME"] = 1.0
    hdu.header["ACQTIME"] = 2458923.123456
    hdu.header["AIRMASS"] = 1.0
    hdu.header["GRISM"] = "J"
    hdu.header["INSTRUME"] = "FIRE"
    hdu.writeto(path)
    return path


def test_state_runs_wavelength_and_tilt_checkpoints(tmp_path):
    flat = make_flat()
    arc = make_arc_image()
    _write_frame(tmp_path / "flat1.fits", flat, "PixFlat", "Flat")
    _write_frame(tmp_path / "flat2.fits", flat * 1.01, "PixFlat", "Flat")
    _write_frame(tmp_path / "arc1.fits", arc, "Arc", "OH")
    _write_frame(tmp_path / "arc2.fits", arc * 1.005, "Arc", "OH")
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})
    state.load_raw_frames(roles=["pixelflat", "arc"])

    state.run_step("prep_calibrations")
    state.run_step("trace_orders")
    state.run_step("calibrate_wavelengths")
    state.run_step("fit_tilts")

    assert state.step == 4
    assert state.step_name == "fit_tilts"
    assert "wavelengths" in state.products
    assert "tilts" in state.products
    assert "wavelength_residuals" in state.plots
    assert "tilt_map" in state.plots

    path = state.save(tmp_path)
    assert path.name == "04_fit_tilts.fits"
    reloaded = ReductionState.load(path)
    np.testing.assert_allclose(
        reloaded.products["wave_soln"], state.products["wave_soln"], equal_nan=True
    )
    assert len(reloaded.products["wavelengths"]) == FIRE_NORDERS

    written = state.save_plots(tmp_path / "qa")
    assert any(p.name.endswith("wavelength_residuals.html") for p in written)
    assert all(p.exists() for p in written)


def test_state_single_order_wavelength_and_tilt_rerun(tmp_path):
    flat = make_flat()
    arc = make_arc_image()
    _write_frame(tmp_path / "flat1.fits", flat, "PixFlat", "Flat")
    _write_frame(tmp_path / "flat2.fits", flat * 1.01, "PixFlat", "Flat")
    _write_frame(tmp_path / "arc1.fits", arc, "Arc", "OH")
    _write_frame(tmp_path / "arc2.fits", arc * 1.005, "Arc", "OH")
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})
    state.load_raw_frames(roles=["pixelflat", "arc"])
    for step in ("prep_calibrations", "trace_orders", "calibrate_wavelengths", "fit_tilts"):
        state.run_step(step)

    wavelengths = state.products["wavelengths"]
    unmasked = FIRE_ORDER_NUMBERS[~np.asarray(wavelengths["masked"], dtype=bool)]
    target = int(unmasked[len(unmasked) // 2])
    index = int(np.where(FIRE_ORDER_NUMBERS == target)[0][0])
    before_wave = state.products["wave_soln"].copy()
    before_tilt = state.products["tilt_coeffs"].copy()

    clone = state.run_step("calibrate_wavelengths", order=target, clone=True)

    assert clone is not state
    assert np.all(np.isfinite(clone.products["wave_soln"][:, index]))
    for other in range(FIRE_NORDERS):
        if other != index:
            np.testing.assert_array_equal(
                clone.products["wave_soln"][:, other], before_wave[:, other]
            )

    tilt_clone = clone.run_step("fit_tilts", order=target, clone=True)
    assert tilt_clone is not clone
    assert np.all(np.isfinite(tilt_clone.products["tilt_coeffs"][index]))
    for other in range(FIRE_NORDERS):
        if other != index:
            np.testing.assert_array_equal(
                tilt_clone.products["tilt_coeffs"][other], before_tilt[other]
            )
    assert state.step == 4

