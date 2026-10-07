"""High-level tests for science processing and sky subtraction.

These exercise the public seams for reduction steps 5-7: the pure
``onfire.core`` science-prep, object-detection, and B-spline sky-modeling
functions, the Step Engines themselves on synthetic FIRE images, and the
``ReductionState`` API that persists the sky-subtracted products to
Multi-Extension FITS with Plotly QA artifacts. Both the Pair Differencing and
Stare Mode workflows are covered.
"""

import numpy as np
import pytest
from astropy.io import fits

from onfire.core import (
    BSplineBasis,
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
    find_objects,
    fit_sky_surface,
    flat_field,
    frame_variance,
    pair_difference,
    prep_calibrations,
    process_science,
    register_engines,
    subtract_sky,
    trace_orders,
)
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable, scan_raw_dir

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
    """Build a synthetic science frame: smooth sky plus Gaussian objects.

    ``objects`` is a list of ``(order, fracpos, amplitude)`` tuples. The flat
    response is applied so ``process_science`` can flat-field it back out.
    """
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
    products = {**prep.products, **traces.products}
    return flat, products, traces.meta


# --------------------------------------------------------------------------- #
# Pure step-engine functions
# --------------------------------------------------------------------------- #


def test_flat_field_removes_flat_response():
    flat = make_flat()
    signal = np.broadcast_to(
        30.0 + np.arange(NP, dtype=float)[None, :] * 0.01, flat.shape
    )
    image = signal * flat

    recovered = flat_field(image, flat)

    np.testing.assert_allclose(recovered, signal, rtol=1e-10)


def test_pair_difference_cancels_common_sky():
    sky = make_flat()
    columns = np.broadcast_to(np.arange(NP, dtype=float)[None, :], sky.shape)
    object_a = 100.0 * np.exp(-0.5 * ((columns - 300.0) / 2.0) ** 2)
    object_b = 100.0 * np.exp(-0.5 * ((columns - 600.0) / 2.0) ** 2)

    difference = pair_difference(sky + object_a, sky + object_b)

    np.testing.assert_allclose(difference, object_a - object_b, atol=1e-9)


def test_frame_variance_scales_with_counts():
    flat = np.full((10, 10), 2.0)
    low = frame_variance(np.full((10, 10), 100.0), flat, gain=2.0, read_noise=3.0)
    high = frame_variance(np.full((10, 10), 400.0), flat, gain=2.0, read_noise=3.0)

    # Poisson term 4x, read-noise term (9/4) fixed, all divided by flat^2.
    np.testing.assert_allclose(low[0, 0], (100.0 / 2.0 + 9.0) / 4.0)
    np.testing.assert_allclose(high[0, 0], (400.0 / 2.0 + 9.0) / 4.0)
    assert np.all(high > low)


def test_bspline_basis_recovers_smooth_function():
    x = np.linspace(0.0, 1000.0, 2000)
    basis = BSplineBasis.uniform(x.min(), x.max(), bspacing=50.0, degree=3)
    design = basis.design(x)
    truth = 3.0 + 0.5 * np.sin(x / 100.0) + 0.01 * x

    coeffs, *_ = np.linalg.lstsq(design, truth, rcond=None)
    rms = np.sqrt(np.mean((design @ coeffs - truth) ** 2))

    assert rms < 1e-3


def test_fit_sky_surface_recovers_known_2d_sky():
    rng = np.random.default_rng(5)
    nspec, nspat = 200, 40
    rows, cols = np.mgrid[0:nspec, 0:nspat]
    pix = rows + 0.2 * cols
    ximg = cols / (nspat - 1)
    truth = (10.0 + 0.1 * pix) * (1.0 + 0.5 * ximg)
    observed = truth + rng.normal(0.0, 0.05, truth.shape)

    fit = fit_sky_surface(
        pix.ravel(),
        ximg.ravel(),
        observed.ravel(),
        weights=np.full(pix.size, 1.0 / 0.05),
        bspacing=25.0,
        degree=3,
        npoly=2,
    )
    model = fit.evaluate(pix.ravel(), ximg.ravel()).reshape(truth.shape)

    assert fit.rms < 0.1
    np.testing.assert_allclose(model, truth, atol=0.2)


# --------------------------------------------------------------------------- #
# process_science engine
# --------------------------------------------------------------------------- #


def test_process_science_pair_difference_workflow():
    _, products, _ = _reference_products()
    frame_a, _, obj_a = make_science(
        products["flat_norm"], [(21, 0.5, 300.0), (20, 0.5, 300.0), (19, 0.5, 300.0)], seed=1
    )
    frame_b, _, obj_b = make_science(
        products["flat_norm"], [(21, 0.35, 300.0), (20, 0.35, 300.0), (19, 0.35, 300.0)], seed=2
    )

    result = process_science(
        {**products, "raw_science": np.stack([frame_a, frame_b])},
        {},
        {"mode": "pair_difference"},
    )

    assert result.meta["science_mode"] == "pair_difference"
    assert result.meta["n_science_combined"] == 1
    science = result.products["science"]
    # Pair differencing cancels the sky, leaving the two nodded objects.
    np.testing.assert_allclose(science, obj_a - obj_b, atol=1e-6)
    assert science.shape == frame_a.shape
    assert result.products["science_var"].shape == science.shape
    assert np.all(result.products["science_ivar"][science != 0] > 0)
    assert "science" in result.plots


def test_process_science_stare_workflow_keeps_sky():
    _, products, _ = _reference_products()
    frame, sky, obj = make_science(products["flat_norm"], [(21, 0.5, 300.0)], seed=3)

    result = process_science(
        {**products, "raw_science": frame[None]}, {}, {"mode": "stare"}
    )

    assert result.meta["science_mode"] == "stare"
    science = result.products["science"]
    # Stare mode preserves the background sky; it is modeled at subtract_sky.
    np.testing.assert_allclose(science, sky + obj, atol=1e-6)
    on_order = np.broadcast_to(
        np.abs(np.arange(NP)[None, :] - FIRE_ORDER_SPAT_POS[0] * (NP - 1)) < 4.0,
        science.shape,
    )
    assert np.median(science[on_order]) > 20.0


def test_process_science_rejects_unknown_mode():
    _, products, _ = _reference_products()
    with pytest.raises(ValueError, match="unknown science mode"):
        process_science({**products, "raw_science": np.zeros((2, NS, NP))}, {}, {"mode": "wobble"})


# --------------------------------------------------------------------------- #
# find_objects engine
# --------------------------------------------------------------------------- #


def test_find_objects_detects_injected_objects():
    _, products, meta = _reference_products()
    objects = [(21, 0.5, 300.0), (20, 0.5, 300.0), (19, 0.5, 300.0)]
    frame, _, _ = make_science(products["flat_norm"], objects, seed=4)
    prepared = process_science(
        {**products, "raw_science": frame[None]}, {}, {"mode": "stare"}
    )
    products.update(prepared.products)

    result = find_objects(products, meta, {})
    table = result.products["objects"]

    assert result.meta["n_objects"] == 3
    assert set(np.asarray(table["order"], dtype=int)) == {19, 20, 21}
    assert np.all(np.asarray(table["snr"], dtype=float) > 5.0)
    assert np.all(np.isfinite(np.asarray(table["spat_pixpos"], dtype=float)))
    assert result.products["object_traces"].shape == (NS, 3)
    assert "object_profiles" in result.plots
    assert "object_traces" in result.plots

    # Object boxes are excluded from the sky mask; other pixels remain usable.
    skymask = result.products["object_skymask"]
    assert not skymask.all()
    assert skymask.sum() > 0
    assert (~skymask).sum() > 0


# --------------------------------------------------------------------------- #
# subtract_sky engine
# --------------------------------------------------------------------------- #


def test_subtract_sky_reduces_stare_background():
    _, products, meta = _reference_products()
    objects = [(21, 0.5, 300.0), (20, 0.5, 300.0), (19, 0.4, 300.0)]
    frame, sky, obj = make_science(products["flat_norm"], objects, seed=5)
    prepared = process_science(
        {**products, "raw_science": frame[None]}, {}, {"mode": "stare"}
    )
    products.update(prepared.products)
    detected = find_objects(products, meta, {})
    products.update(detected.products)

    result = subtract_sky(products, meta, {})
    model = result.products["sky_model"]
    residual = result.products["science_sky"]
    skymask = products["object_skymask"]
    modeled = (model != 0) & skymask

    assert modeled.sum() > 1000
    before = np.sqrt(np.mean((sky + obj)[modeled] ** 2))
    after = np.sqrt(np.mean(residual[modeled] ** 2))
    assert after < 0.05 * before
    assert after < 1.0
    assert result.meta["sky_residual_rms"] < 0.1
    assert result.products["science_sky_var"].shape == frame.shape
    assert "sky_model" in result.plots


def test_subtract_sky_pair_differenced_data_is_near_zero():
    _, products, meta = _reference_products()
    frame_a, _, _ = make_science(products["flat_norm"], [(21, 0.5, 300.0)], seed=6)
    frame_b, _, _ = make_science(products["flat_norm"], [(21, 0.35, 300.0)], seed=7)
    prepared = process_science(
        {**products, "raw_science": np.stack([frame_a, frame_b])},
        {},
        {"mode": "pair_difference"},
    )
    products.update(prepared.products)
    detected = find_objects(products, meta, {})
    products.update(detected.products)

    result = subtract_sky(products, meta, {})

    # The sky was already cancelled at the exposure level, so the model is flat.
    assert result.meta["sky_residual_rms"] < 1.0


# --------------------------------------------------------------------------- #
# ReductionState end-to-end
# --------------------------------------------------------------------------- #


def _write_frame(path, data, obstype, name):
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.header["OBSTYPE"] = obstype
    hdu.header["OBJECT"] = name
    hdu.header["EXPTIME"] = 1.0
    hdu.header["ACQTIME"] = 2458923.123456
    hdu.header["AIRMASS"] = 1.0
    hdu.header["GRISM"] = "J"
    hdu.header["INSTRUME"] = "FIRE"
    hdu.writeto(path)
    return path


def test_checkpoint_roundtrips_object_table_and_sky_products(tmp_path):
    _, products, meta = _reference_products()
    frame, _, _ = make_science(
        products["flat_norm"], [(21, 0.5, 300.0), (20, 0.5, 300.0)], seed=8
    )
    prepared = process_science(
        {**products, "raw_science": frame[None]}, {}, {"mode": "stare"}
    )
    products.update(prepared.products)
    detected = find_objects(products, meta, {})
    products.update(detected.products)
    products.update(subtract_sky(products, meta, {}).products)

    state = ReductionState(FrameTable.from_frames([]), {})
    state.products.update(products)
    state.step = 7
    path = state.save(tmp_path)
    reloaded = ReductionState.load(path)

    objects = reloaded.products["objects"]
    assert len(objects) == 2
    assert objects["trace_coeff"].shape[1] == 4
    np.testing.assert_allclose(
        reloaded.products["science_sky"], products["science_sky"], equal_nan=True
    )
    np.testing.assert_allclose(
        reloaded.products["science_sky_var"],
        products["science_sky_var"],
        equal_nan=True,
    )
    assert reloaded.products["object_skymask"].dtype == bool


def test_state_runs_science_processing_and_checkpoints(tmp_path):
    flat = make_flat()
    _write_frame(tmp_path / "flat1.fits", flat, "PixFlat", "Flat")
    _write_frame(tmp_path / "flat2.fits", flat * 1.01, "PixFlat", "Flat")
    _write_frame(tmp_path / "sci1.fits", flat * 0.5, "Science", "OBJ")
    _write_frame(tmp_path / "sci2.fits", flat * 0.49, "Science", "OBJ")
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})
    state.load_raw_frames(roles=["pixelflat", "science"])

    for step in ("prep_calibrations", "trace_orders", "process_science", "find_objects", "subtract_sky"):
        state.run_step(step, params={"mode": "pair_difference"})

    assert state.step == 7
    assert state.step_name == "subtract_sky"
    assert state.products["science"].shape == flat.shape
    assert "objects" in state.products
    assert "sky_model" in state.products
    assert "science_sky" in state.products
    assert {"science", "object_profiles", "sky_model"} <= set(state.plots)

    path = state.save(tmp_path)
    assert path.name == "07_subtract_sky.fits"
    reloaded = ReductionState.load(path)
    np.testing.assert_allclose(
        reloaded.products["science_sky"], state.products["science_sky"], equal_nan=True
    )
    assert reloaded.products["objects"].colnames[0] == "order"
    assert reloaded.products["object_skymask"].dtype == bool

    written = state.save_plots(tmp_path / "qa")
    assert any(p.name.endswith("sky_model.html") for p in written)
    assert all(p.exists() for p in written)


def test_state_reads_science_mode_from_parameter_config():
    flat = make_flat()
    state = ReductionState(FrameTable.from_frames([]), {"mode": "stare"})
    state.products["flat_norm"] = flat
    state.products["raw_science"] = np.stack([flat * 0.5])

    state.run_step("process_science")

    assert state.meta["science_mode"] == "stare"
