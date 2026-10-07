"""High-level tests for 1D spectral extraction (reduction step 8).

These exercise the public seams for extraction: the pure ``onfire.core`` boxcar,
profile-fitting, and Horne optimal-extraction functions, the ``extract_1d`` Step
Engine on synthetic FIRE images, and the ``ReductionState`` API that persists
the extracted multi-order spectra to Multi-Extension FITS with Plotly QA
artifacts. Flux conservation, variance propagation, and profile weighting are
covered directly, alongside interactive single-order re-extraction.
"""

import numpy as np
import pytest

from onfire.core import (
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
    boxcar_extract,
    build_profile_image,
    collapse_object_profile,
    extract_1d,
    find_objects,
    fit_object_profile,
    optimal_extract,
    prep_calibrations,
    process_science,
    register_engines,
    subtract_sky,
    trace_orders,
)
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable

NS, NP = 150, 1000
SIGMA = 6.0


@pytest.fixture(autouse=True)
def _engines():
    clear_step_engines()
    register_engines()
    yield
    clear_step_engines()


def make_flat(nspec=NS, nspat=NP, *, sigma=SIGMA, amplitude=200.0, background=20.0, seed=0):
    """Build a synthetic FIRE-like flat of separated Gaussian order stripes."""
    rng = np.random.default_rng(seed)
    centers = FIRE_ORDER_SPAT_POS * (nspat - 1)
    cols = np.arange(nspat, dtype=float)[None, :]
    flat = np.full((nspec, nspat), background, dtype=float)
    for center in centers:
        flat += amplitude * np.exp(-0.5 * ((cols - center) / sigma) ** 2)
    flat += rng.normal(0.0, 0.5, flat.shape)
    return flat


def make_science(flat_norm, objects, *, nspec=NS, nspat=NP, seed=1):
    """Build a synthetic science frame: smooth sky plus Gaussian objects."""
    rng = np.random.default_rng(seed)
    centers = FIRE_ORDER_SPAT_POS * (nspat - 1)
    rows = np.arange(nspec, dtype=float)[:, None]
    cols = np.arange(nspat, dtype=float)[None, :]
    sky = np.zeros((nspec, nspat))
    for center in centers:
        region = 12.0
        ximg = (cols - (center - region)) / (2 * region)
        on = np.abs(cols - center) < region
        level = 40.0 * (1 + 0.1 * np.sin(rows / 30.0))
        shape = 1.0 + 0.3 * (ximg - 0.5) + 0.15 * (ximg - 0.5) ** 2
        sky += np.where(on, level * shape, 0.0)
    obj = np.zeros((nspec, nspat))
    for order, fracpos, amplitude in objects:
        index = int(np.where(FIRE_ORDER_NUMBERS == order)[0][0])
        center = centers[index]
        obj_center = (center - 6.0) + fracpos * 12.0
        obj += (
            amplitude
            * np.exp(-0.5 * ((cols - obj_center) / 1.5) ** 2)
            * (1 + 0.2 * np.sin(rows / 25.0))
        )
    return (sky + obj) * flat_norm, sky, obj


def _reference_products():
    """Calibrated products with well-traced orders for engine-level tests."""
    flat = make_flat()
    prep = prep_calibrations({"raw_flats": flat[None]}, {}, {})
    traces = trace_orders({"flat_norm": flat}, {}, {"ncoeff": 4})
    return flat, {**prep.products, **traces.products}, traces.meta


def _extractable_products(objects=((21, 0.5, 300.0), (20, 0.5, 300.0), (19, 0.5, 300.0)), seed=11):
    """Run the pipeline through subtract_sky and return products plus meta."""
    _, products, meta = _reference_products()
    frame, _, _ = make_science(products["flat_norm"], list(objects), seed=seed)
    prepared = process_science({**products, "raw_science": frame[None]}, {}, {"mode": "stare"})
    products.update(prepared.products)
    products.update(find_objects(products, meta, {}).products)
    products.update(subtract_sky(products, meta, {}).products)
    return products, meta


def _gaussian_object(nspec=40, nspat=60, *, centre=30.0, sigma=2.0, flux=500.0, ivar=1.0):
    """Return a detector image holding one known-flux Gaussian object."""
    trace = np.full(nspec, float(centre))
    cols = np.arange(nspat, dtype=float)[None, :]
    profile = np.exp(-0.5 * ((cols - trace[:, None]) / sigma) ** 2)
    profile /= profile.sum(axis=1, keepdims=True)
    image = flux * profile
    ivar_image = np.full((nspec, nspat), float(ivar))
    mask = np.ones((nspec, nspat), dtype=bool)
    return image, ivar_image, trace, profile, mask


# --------------------------------------------------------------------------- #
# Pure step-engine functions
# --------------------------------------------------------------------------- #


def test_boxcar_extract_conserves_object_flux():
    image, ivar, trace, _, mask = _gaussian_object()

    result = boxcar_extract(image, ivar, trace, 10.0, mask=mask)

    np.testing.assert_allclose(result.flux, 500.0, rtol=1e-3)
    np.testing.assert_allclose(result.var, 21.0, rtol=1e-6)
    np.testing.assert_allclose(result.ivar, 1.0 / 21.0, rtol=1e-6)
    assert np.all(result.npix == 21)
    assert np.all(result.mask)


def test_optimal_extract_conserves_flux_and_matches_horne_variance():
    image, ivar, trace, profile, mask = _gaussian_object()

    result = optimal_extract(image, ivar, profile, mask=mask)

    expected_var = 1.0 / np.sum(profile**2, axis=1)
    np.testing.assert_allclose(result.flux, 500.0, rtol=1e-9)
    np.testing.assert_allclose(result.var, expected_var, rtol=1e-9)
    np.testing.assert_allclose(result.ivar, 1.0 / expected_var, rtol=1e-9)
    assert np.all(result.mask)
    assert np.all(result.frac_use > 0.99)
    np.testing.assert_allclose(result.chi2, 0.0, atol=1e-12)


def test_optimal_extraction_beats_boxcar_for_peaked_profile():
    image, ivar, trace, profile, mask = _gaussian_object()

    box = boxcar_extract(image, ivar, trace, 10.0, mask=mask)
    optimal = optimal_extract(image, ivar, profile, mask=mask)

    # Optimal weighting of a peaked profile lowers the variance versus a boxcar.
    assert np.all(optimal.var < box.var)


def test_optimal_extract_rejects_cosmic_rays():
    image, ivar, trace, profile, mask = _gaussian_object()
    image[10, 30] += 5000.0

    result = optimal_extract(image, ivar, profile, mask=mask, sigclip=5.0, niter=3)

    assert result.cr_mask[10, 30]
    assert result.cr_mask.sum() == 1
    np.testing.assert_allclose(result.flux, 500.0, rtol=1e-6)


def test_fit_object_profile_recovers_gaussian_width():
    image, ivar, trace, _, mask = _gaussian_object(sigma=2.0)

    offsets, template = collapse_object_profile(image, ivar, trace, half_width=8, mask=mask)
    fit = fit_object_profile(offsets, template)

    assert fit.kind == "gaussian"
    np.testing.assert_allclose(fit.sigma, 2.0, atol=0.05)
    np.testing.assert_allclose(fit.fwhm, 2.3548 * 2.0, rtol=0.05)


def test_fit_object_profile_recovers_offcentre_gaussian():
    offsets = np.arange(60, dtype=float)
    values = np.exp(-0.5 * ((offsets - 35.0) / 1.5) ** 2)

    fit = fit_object_profile(offsets, values)

    assert fit.kind == "gaussian"
    np.testing.assert_allclose(fit.coeffs[1], 35.0, atol=0.05)
    np.testing.assert_allclose(fit.sigma, 1.5, atol=0.05)


def test_polynomial_profile_model_conserves_flux():
    image, ivar, trace, _, mask = _gaussian_object()

    offsets, template = collapse_object_profile(image, ivar, trace, half_width=8, mask=mask)
    fit = fit_object_profile(offsets, template, model="polynomial", order=4)
    profile = build_profile_image(fit, trace, image.shape, half_width=8, mask=mask)
    result = optimal_extract(image, ivar, profile, mask=mask)

    assert fit.kind == "polynomial"
    np.testing.assert_allclose(result.flux, 500.0, rtol=0.05)


def test_optimal_extract_propagates_non_uniform_variance():
    image, ivar, trace, profile, mask = _gaussian_object(ivar=2.0)
    ivar = ivar * np.linspace(1.0, 2.0, image.shape[1])[None, :]

    result = optimal_extract(image, ivar, profile, mask=mask)

    expected = np.sum(ivar * profile**2, axis=1) / np.sum(profile, axis=1)
    np.testing.assert_allclose(result.ivar, expected, rtol=1e-9)


# --------------------------------------------------------------------------- #
# extract_1d engine
# --------------------------------------------------------------------------- #


def test_extract_1d_extracts_all_objects():
    products, meta = _extractable_products()

    result = extract_1d(products, meta, {})

    objects = products["objects"]
    table = result.products["extractions"]
    assert len(table) == len(objects)
    assert list(table.colnames)[:5] == ["order", "obj_id", "snr", "fwhm", "box_radius"]
    assert result.meta["n_objects_extracted"] == 3
    assert result.meta["n_orders_extracted"] == 3
    assert result.products["opt_flux"].shape == (NS, 3)
    assert result.products["boxcar_flux"].shape == (NS, 3)
    assert result.products["opt_mask"].dtype == bool
    assert result.products["extraction_cr_mask"].shape == (NS, NP)


def test_extract_1d_optimal_weights_and_propagates_variance():
    products, meta = _extractable_products()

    result = extract_1d(products, meta, {})
    optimal = result.products["opt_flux"]
    boxcar = result.products["boxcar_flux"]
    opt_var = result.products["opt_var"]
    opt_ivar = result.products["opt_ivar"]
    opt_mask = result.products["opt_mask"]

    # Where both extractions are valid they measure the same object.
    both = opt_mask & np.isfinite(boxcar)
    assert both.any()
    ratio = optimal[both] / boxcar[both]
    assert np.all(ratio > 0.5)

    # Inverse variance is the reciprocal of the propagated variance.
    np.testing.assert_allclose(opt_ivar[opt_mask], 1.0 / opt_var[opt_mask], rtol=1e-6)

    # Optimal weighting cannot lose S/N relative to a boxcar on average.
    valid = opt_mask & np.isfinite(result.products["boxcar_var"])
    assert np.nanmedian(opt_var[valid]) <= np.nanmedian(result.products["boxcar_var"][valid])


def test_extract_1d_generates_qa_figures():
    products, meta = _extractable_products()

    result = extract_1d(products, meta, {})

    assert {"extraction_spectra", "extraction_noise", "extraction_profiles"} <= set(
        result.plots
    )


def test_extract_1d_requires_sky_subtracted_image():
    with pytest.raises(ValueError, match="science_sky"):
        extract_1d({}, {}, {})


def test_extract_1d_single_order_reextraction_preserves_other_orders():
    products, meta = _extractable_products()
    first = extract_1d(products, meta, {})
    products.update(first.products)

    second = extract_1d(products, meta, {"order": 21, "box_radius": 3.0})

    orders = np.asarray(products["objects"]["order"], dtype=int)
    target = orders == 21
    other = ~target

    # The targeted order was re-extracted with the new aperture...
    assert not np.allclose(
        first.products["boxcar_flux"][:, target],
        second.products["boxcar_flux"][:, target],
        equal_nan=True,
    )
    # ...while every other order is byte-for-byte preserved.
    np.testing.assert_allclose(
        second.products["opt_flux"][:, other],
        first.products["opt_flux"][:, other],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        second.products["boxcar_flux"][:, other],
        first.products["boxcar_flux"][:, other],
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# ReductionState end-to-end
# --------------------------------------------------------------------------- #


def test_state_runs_extraction_and_checkpoints(tmp_path):
    products, meta = _extractable_products()
    state = ReductionState(FrameTable.from_frames([]), {})
    state.products.update(products)
    state.step = 7

    state.run_step("extract_1d")

    assert state.step == 8
    assert state.step_name == "extract_1d"
    assert "extractions" in state.products
    assert "opt_flux" in state.products
    assert {"extraction_spectra", "extraction_noise", "extraction_profiles"} <= set(
        state.plots
    )

    path = state.save(tmp_path)
    assert path.name == "08_extract_1d.fits"
    reloaded = ReductionState.load(path)
    np.testing.assert_allclose(
        reloaded.products["opt_flux"], state.products["opt_flux"], equal_nan=True
    )
    assert reloaded.products["extraction_cr_mask"].dtype == bool
    assert reloaded.products["extractions"].colnames[0] == "order"
    np.testing.assert_allclose(
        np.asarray(reloaded.products["extractions"]["opt_flux"]),
        np.asarray(state.products["extractions"]["opt_flux"]),
        equal_nan=True,
    )
