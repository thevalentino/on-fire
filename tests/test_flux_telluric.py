"""High-level tests for flux calibration and telluric correction (steps 9-10).

These exercise the public seams for spectrophotometric flux calibration and
telluric absorption removal: the pure ``onfire.core.flux`` and
``onfire.core.telluric`` functions, the ``flux_calibrate`` and
``telluric_correct`` Step Engines, and the ``ReductionState`` API including the
interactive ``compare_telluric`` selector and Multi-Extension FITS persistence.
"""

import numpy as np
import pytest
from astropy.table import Table

from onfire.core import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    SensitivityFunction,
    TelluricCorrection,
    apply_sensitivity,
    apply_telluric,
    compute_zeropoint,
    counts_to_nlam,
    derive_sensitivity,
    empirical_transmission,
    fit_sensitivity_function,
    fit_telluric_grid,
    fit_telluric_pca,
    flux_calibrate,
    nlam_to_flam,
    pca_transmission,
    register_engines,
    telluric_correct,
)
from onfire.core.flux import ZP_UNIT_CONST
from onfire.data import load_a0v_template
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable

NS = 300


@pytest.fixture(autouse=True)
def _engines():
    clear_step_engines()
    register_engines()
    yield
    clear_step_engines()


def _synthetic_sensitivity(nspec=NS):
    """Return a smooth synthetic wavelength grid and zeropoint curve."""
    wave = np.linspace(1.0e4, 2.4e4, nspec)
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    zeropoint = 20.5 + 0.6 * np.sin(np.pi * x) + 0.2 * x
    return wave, zeropoint


def _smooth_spectrum(wave, scale=4.0e-13, index=-2.5):
    """Return a smooth, strictly positive standard-star-like spectrum."""
    return scale * (wave / 1.5e4) ** index


def _vega_like(nspec=NS):
    template = load_a0v_template()
    wave = np.linspace(template["wave"].min() + 1.0, template["wave"].max() - 1.0, nspec)
    flux = np.interp(wave, template["wave"], template["flux"])
    return wave, flux


def _grid_components(nwave=NS, ncomp=4, seed=3):
    """Build a synthetic PCA atmospheric grid on a log wavelength grid."""
    rng = np.random.default_rng(seed)
    wave = np.logspace(np.log10(1.0e4), np.log10(2.4e4), nwave)
    x = np.linspace(-1.0, 1.0, nwave)

    def line(centre, width):
        return np.exp(-0.5 * ((x - centre) / width) ** 2)

    components = np.zeros((ncomp, nwave))
    components[0] = 0.05 + 0.6 * line(0.2, 0.05) + 0.5 * line(-0.4, 0.06) + 0.4 * line(0.7, 0.08)
    components[1] = 0.3 * line(0.2, 0.05) - 0.1 * line(-0.4, 0.06)
    components[2] = -0.2 * line(-0.4, 0.06) + 0.15 * line(0.7, 0.08)
    components[3] = 0.1 * line(0.7, 0.08)
    components += rng.normal(0.0, 1e-4, components.shape)
    return wave, components


# --------------------------------------------------------------------------- #
# Pure flux-calibration functions
# --------------------------------------------------------------------------- #


def test_zp_unit_const_matches_reference_value():
    assert ZP_UNIT_CONST == pytest.approx(40.092117379602044, rel=1e-9)


def test_counts_to_nlam_scales_and_propagates_variance():
    wave = np.arange(1000.0, 2000.0, 10.0)
    counts = np.full(wave.size, 100.0)
    counts_ivar = np.full(wave.size, 0.04)

    nlam, nlam_ivar = counts_to_nlam(wave, counts, 10.0, counts_ivar=counts_ivar)

    np.testing.assert_allclose(nlam, 1.0)
    # dlambda^2 * ivar * exptime^2
    np.testing.assert_allclose(nlam_ivar, 10.0**2 * 0.04 * 10.0**2)


def test_compute_zeropoint_recovers_synthetic_zeropoint():
    wave, zeropoint = _synthetic_sensitivity()
    factor = nlam_to_flam(wave, zeropoint)
    standard = _smooth_spectrum(wave)
    nlam = standard / factor

    recovered, good = compute_zeropoint(wave, nlam, standard)

    assert good.all()
    np.testing.assert_allclose(recovered, zeropoint, atol=1e-9)


def test_fit_sensitivity_function_recovers_polynomial_zeropoint():
    wave = np.linspace(1.0e4, 2.4e4, NS)
    x = 2.0 * (wave - wave.min()) / (wave.max() - wave.min()) - 1.0
    zeropoint = 20.5 + 0.8 * x - 0.4 * x**2

    sensfunc = fit_sensitivity_function(wave, zeropoint, order=4)

    assert isinstance(sensfunc, SensitivityFunction)
    assert sensfunc.order == 4
    np.testing.assert_allclose(sensfunc.evaluate_zeropoint(wave), zeropoint, atol=1e-8)


def test_derive_and_apply_sensitivity_round_trips_flux():
    wave, zeropoint = _synthetic_sensitivity()
    factor = nlam_to_flam(wave, zeropoint)
    sampling = np.abs(np.gradient(wave))
    exptime = 300.0
    standard = _smooth_spectrum(wave)
    standard_counts = standard / factor * exptime * sampling

    sensfunc = derive_sensitivity(
        wave, standard_counts, standard, exptime=exptime, sampling=sampling, order=8
    )

    science = _smooth_spectrum(wave, scale=9.0e-14)
    science_counts = science / factor * exptime * sampling
    science_var = np.abs(science_counts) * 0.1
    flux, flux_var, flux_ivar, mask = apply_sensitivity(
        sensfunc,
        wave,
        science_counts,
        exptime=exptime,
        counts_var=science_var,
        sampling=sampling,
    )

    assert mask.all()
    np.testing.assert_allclose(flux[mask], science[mask], rtol=5e-3)
    np.testing.assert_allclose(flux_ivar[mask], 1.0 / flux_var[mask], rtol=1e-6)
    # Variance scales with the square of the linear counts-to-flux conversion.
    np.testing.assert_allclose(
        flux_var[mask],
        (science_var * (factor / (exptime * sampling)) ** 2)[mask],
        rtol=5e-3,
    )


# --------------------------------------------------------------------------- #
# Pure telluric functions
# --------------------------------------------------------------------------- #


def _transmission_with_lines(wave, *, floor=0.15):
    """Return a transmission curve with a few Gaussian absorption features."""
    transmission = np.ones_like(wave)
    for centre, width, depth in ((1.25e4, 60.0, 0.7), (1.6e4, 90.0, 0.5), (2.1e4, 120.0, 0.6)):
        transmission -= depth * np.exp(-0.5 * ((wave - centre) / width) ** 2)
    return np.clip(transmission, floor, 1.0)


def test_empirical_transmission_recovers_known_absorption():
    wave = np.linspace(1.0e4, 2.4e4, NS)
    a0v = _smooth_spectrum(wave)
    response = 0.5 + 0.1 * np.sin((wave - 1.0e4) / 8.0e3)
    truth = _transmission_with_lines(wave)
    standard_flux = a0v * response * truth

    correction = empirical_transmission(wave, standard_flux, a0v)

    assert isinstance(correction, TelluricCorrection)
    assert correction.method == "empirical"
    assert np.all(correction.transmission <= 1.0)
    # Deep absorption features must survive the continuum removal.
    dip = truth < 0.5
    assert dip.any()
    np.testing.assert_allclose(
        correction.transmission[dip], truth[dip], atol=0.05
    )


def test_apply_telluric_recovers_intrinsic_flux_and_var():
    wave = np.linspace(1.0e4, 2.4e4, NS)
    truth = _transmission_with_lines(wave)
    intrinsic = _smooth_spectrum(wave, scale=1.0e-13)
    observed = intrinsic * truth
    variance = 0.01 * observed**2
    correction = TelluricCorrection(wave, truth, np.ones_like(wave, dtype=bool), "grid")

    flux, out_var, out_ivar, mask = apply_telluric(
        wave, observed, correction, variance=variance
    )

    assert mask.all()
    np.testing.assert_allclose(flux, intrinsic, rtol=1e-6)
    np.testing.assert_allclose(out_var, variance / truth**2, rtol=1e-6)
    np.testing.assert_allclose(out_ivar[mask], 1.0 / out_var[mask], rtol=1e-6)


def test_pca_transmission_round_trips_known_coefficients():
    wave, components = _grid_components()
    coeffs = np.array([0.7, -0.4, 0.3])

    transmission = pca_transmission(wave, components, coeffs)
    correction = fit_telluric_pca(wave, transmission, components, ncoeff=3)

    assert correction.method == "grid"
    np.testing.assert_allclose(correction.coeffs, coeffs, atol=1e-3)
    np.testing.assert_allclose(correction.transmission, transmission, atol=1e-6)


def test_fit_telluric_grid_removes_absorption_from_science():
    wave, components = _grid_components()
    truth = pca_transmission(wave, components, np.array([0.6, -0.3, 0.4]))
    continuum = 1.0e-13 * (wave / 1.5e4) ** -2.0
    flux = continuum * truth

    correction = fit_telluric_grid(wave, flux, components, object_order=4, ncoeff=3)

    assert correction.method == "grid"
    # The recovered transmission tracks the injected absorption.
    deep = truth < 0.7
    assert deep.any()
    assert np.corrcoef(correction.transmission, truth)[0, 1] > 0.9
    np.testing.assert_allclose(
        correction.transmission[deep], truth[deep], atol=0.05
    )


# --------------------------------------------------------------------------- #
# flux_calibrate and telluric_correct Step Engines
# --------------------------------------------------------------------------- #

ORDERS = (21, 20, 19)


def _base_products(nspec=NS, orders=ORDERS):
    """Build synthetic wave_soln, objects, A0V template, and instrument factor."""
    wave = np.linspace(1.0e4, 2.4e4, nspec)
    sampling = np.abs(np.gradient(wave))
    wave_soln = np.tile(wave[:, None], (1, FIRE_NORDERS))
    template = load_a0v_template()
    a0v = np.interp(wave, template["wave"], template["flux"])
    x = np.linspace(-1.0, 1.0, nspec)
    factor = nlam_to_flam(wave, 20.5 + 0.6 * np.sin(np.pi * x))
    objects = Table(
        {
            "order": np.asarray(orders, dtype=np.int16),
            "obj_id": np.arange(len(orders), dtype=np.int32),
        }
    )
    return wave, sampling, wave_soln, a0v, factor, objects


def _science_spectrum(wave, scale=1.0e-13):
    return scale * (wave / 1.5e4) ** -2.0


def test_flux_calibrate_recovers_known_science_flux():
    wave, sampling, wave_soln, a0v, factor, objects = _base_products()
    standard_counts = np.tile((a0v / factor * sampling)[:, None], (1, FIRE_NORDERS))
    science = _science_spectrum(wave)
    opt_flux = np.tile((science / factor * sampling)[:, None], (1, len(ORDERS)))
    opt_var = np.abs(opt_flux) * 0.01
    products = {
        "wave_soln": wave_soln,
        "objects": objects,
        "standard_flux": standard_counts,
        "opt_flux": opt_flux,
        "opt_var": opt_var,
    }

    result = flux_calibrate(products, {}, {"sensfunc_order": 8})

    assert result.meta["n_orders_flux_calibrated"] == FIRE_NORDERS
    assert result.meta["n_objects_flux_calibrated"] == len(ORDERS)
    assert not np.any(np.asarray(result.products["sensfunc"]["masked"]))
    valid = result.products["flux_mask"]
    assert valid.any()
    np.testing.assert_allclose(
        result.products["flux"][valid],
        np.broadcast_to(science[:, None], opt_flux.shape)[valid],
        rtol=1e-3,
    )
    assert {"sensitivity", "flux_calibration"} <= set(result.plots)


def _telluric_products(truth=None):
    """Products for telluric_correct with an A0V standard and a grid."""
    wave, sampling, wave_soln, a0v, factor, objects = _base_products()
    _, components = _grid_components(nwave=NS, ncomp=4)
    if truth is None:
        truth = pca_transmission(wave, components, np.array([0.6, -0.3, 0.4]))
    standard_counts = np.tile((a0v * truth / factor * sampling)[:, None], (1, FIRE_NORDERS))
    standard_true = np.tile(a0v[:, None], (1, FIRE_NORDERS))
    science = _science_spectrum(wave)
    flux = np.tile((science * truth)[:, None], (1, len(ORDERS)))
    flux_var = 0.01 * flux**2
    opt_flux = np.tile((science * truth / factor * sampling)[:, None], (1, len(ORDERS)))
    opt_var = np.abs(opt_flux) * 0.01
    products = {
        "wave_soln": wave_soln,
        "objects": objects,
        "standard_flux": standard_counts,
        "standard_true": standard_true,
        "telluric_pca": components,
        "opt_flux": opt_flux,
        "opt_var": opt_var,
        "flux": flux,
        "flux_var": flux_var,
    }
    return products, wave, truth, science


def test_telluric_correct_empirical_removes_absorption():
    wave, _, _, _, _, _ = _base_products()
    truth = _transmission_with_lines(wave, floor=0.2)
    products, wave, truth, science = _telluric_products(truth=truth)

    result = telluric_correct(products, {}, {})

    assert result.meta["telluric_method"] == "empirical"
    transmission = result.products["telluric_transmission"]
    deep = truth < 0.6
    assert deep.any()
    assert np.corrcoef(transmission[:, 0], truth)[0, 1] > 0.95
    np.testing.assert_allclose(transmission[:, 0], truth, atol=0.12)
    valid = result.products["telluric_mask"]
    assert valid.any()
    intrinsic = np.broadcast_to(science[:, None], result.products["telluric_flux"].shape)
    np.testing.assert_allclose(
        result.products["telluric_flux"][valid], intrinsic[valid], rtol=0.15
    )


def test_telluric_correct_grid_method_uses_pca():
    products, wave, truth, science = _telluric_products()

    result = telluric_correct(products, {}, {"method": "grid", "telluric_ncoeff": 3})

    assert result.meta["telluric_method"] == "grid"
    transmission = result.products["telluric_transmission"]
    assert np.corrcoef(transmission[:, 0], truth)[0, 1] > 0.9
    np.testing.assert_allclose(transmission[:, 0], truth, atol=0.08)


def test_telluric_correct_auto_falls_back_to_grid_without_standard():
    products, _, _, _ = _telluric_products()
    products.pop("standard_flux")
    products.pop("standard_true")

    result = telluric_correct(products, {}, {})

    assert result.meta["telluric_method"] == "grid"
    assert result.meta["telluric_available_empirical"] is False
    assert result.meta["telluric_available_grid"] is True


def test_telluric_correct_requires_a_correction_source():
    products, _, _, _ = _telluric_products()
    products.pop("standard_flux")
    products.pop("standard_true")
    products.pop("telluric_pca")

    with pytest.raises(ValueError, match="neither a standard star nor a telluric"):
        telluric_correct(products, {}, {})


def test_telluric_correct_computes_both_candidates():
    products, _, _, _ = _telluric_products()

    result = telluric_correct(products, {}, {"method": "empirical"})

    assert result.meta["telluric_available_empirical"] is True
    assert result.meta["telluric_available_grid"] is True
    assert "telluric_empirical_transmission" in result.products
    assert "telluric_grid_transmission" in result.products


def test_telluric_correct_requires_flux_calibrated_input():
    with pytest.raises(ValueError, match="flux"):
        telluric_correct({"wave_soln": np.zeros((NS, FIRE_NORDERS)), "objects": Table({"order": [], "obj_id": []})}, {}, {})


# --------------------------------------------------------------------------- #
# ReductionState interactive selection and persistence
# --------------------------------------------------------------------------- #


def test_state_compare_telluric_renders_and_commits(tmp_path):
    products, _, _, _ = _telluric_products()
    state = ReductionState(FrameTable.from_frames([]), {})
    state.products.update(products)
    state.step = 9

    state.run_step("telluric_correct")

    assert state.meta["telluric_method"] == "empirical"
    empirical_flux = np.asarray(state.products["telluric_empirical_flux"]).copy()
    grid_flux = np.asarray(state.products["telluric_grid_flux"]).copy()

    figure = state.compare_telluric()
    assert figure is not None
    assert "telluric_comparison" in state.plots
    # Rendering alone does not change the committed correction.
    np.testing.assert_allclose(state.products["telluric_flux"], empirical_flux, equal_nan=True)

    state.compare_telluric(commit="grid")

    assert state.meta["telluric_method"] == "grid"
    assert str(state.products["telluric"]["method"][0]) == "grid"
    np.testing.assert_allclose(state.products["telluric_flux"], grid_flux, equal_nan=True)


def test_state_flux_and_telluric_checkpoint_round_trip(tmp_path):
    products, _, _, _ = _telluric_products()
    state = ReductionState(FrameTable.from_frames([]), {})
    state.products.update(products)
    state.step = 8

    state.run_step("flux_calibrate")
    assert state.step == 9
    state.run_step("telluric_correct")
    assert state.step == 10

    path = state.save(tmp_path)
    assert path.name == "10_telluric_correct.fits"
    reloaded = ReductionState.load(path)

    np.testing.assert_allclose(
        reloaded.products["flux"], state.products["flux"], equal_nan=True
    )
    np.testing.assert_allclose(
        reloaded.products["telluric_flux"],
        state.products["telluric_flux"],
        equal_nan=True,
    )
    assert reloaded.products["telluric"].colnames[0] == "order"
    assert reloaded.products["sensfunc"].colnames[0] == "order"
    assert reloaded.meta["telluric_method"] == "empirical"
