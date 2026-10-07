"""High-level tests for calibration prep and order edge tracing.

These exercise the public seams for reduction steps 1 and 2: the pure
``onfire.core`` Step Engines with synthetic detector images, and the
``ReductionState`` API (``load_raw_frames``, ``run_step``, ``save``, ``load``,
``save_plots``) that wires them into Multi-Extension FITS checkpoints and
Plotly QA artifacts.
"""

import numpy as np
import pytest
from astropy.io import fits

from onfire.core import (
    FIRE_NORDERS,
    FIRE_ORDER_SPAT_POS,
    build_bad_pixel_mask,
    combine_flats,
    detect_order_edges,
    evaluate_edge_trace,
    register_engines,
    trace_orders,
)
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable, scan_raw_dir


@pytest.fixture(autouse=True)
def _engines():
    clear_step_engines()
    register_engines()
    yield
    clear_step_engines()


def make_flat(nspec=160, nspat=600, *, slope=0.0, sigma=3.0, amplitude=200.0,
              background=20.0, hot_pixel=None, seed=0):
    """Build a synthetic FIRE-like detector flat of separated echelle orders.

    Each order is a Gaussian stripe centred on the canonical FIRE order
    position, optionally shifted linearly with spectral row (``slope``) to
    imitate the physical tilt of the orders. Also returns the true left/right
    edge positions at the central row.
    """
    rng = np.random.default_rng(seed)
    centers = FIRE_ORDER_SPAT_POS * (nspat - 1)
    rows = np.arange(nspec, dtype=float)[:, None]
    cols = np.arange(nspat, dtype=float)[None, :]
    flat = np.full((nspec, nspat), background, dtype=float)
    for center in centers:
        shifted = center + slope * (rows - nspec / 2.0)
        flat += amplitude * np.exp(-0.5 * ((cols - shifted) / sigma) ** 2)
    flat += rng.normal(0.0, 0.5, size=flat.shape)
    if hot_pixel is not None:
        flat[hot_pixel] = 1.0e6
    left = centers - sigma
    right = centers + sigma
    return flat, left, right


def _write_flat(path, data):
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.header["OBSTYPE"] = "PixFlat"
    hdu.header["OBJECT"] = "Flat"
    hdu.header["EXPTIME"] = 1.0
    hdu.header["ACQTIME"] = 2458923.123456
    hdu.header["AIRMASS"] = 1.0
    hdu.header["GRISM"] = "J"
    hdu.header["INSTRUME"] = "FIRE"
    hdu.writeto(path)
    return path


def test_combine_flats_averages_identical_exposures():
    image = np.full((20, 30), 100.0)
    flats = np.stack([image, image, image])

    combined, variance, good, nused = combine_flats(flats)

    np.testing.assert_allclose(combined, 100.0)
    np.testing.assert_allclose(variance, 100.0 / 3.0)
    assert good.all()
    assert (nused == 3).all()


def test_combine_flats_rejects_outlier_exposure():
    base = np.full((5, 20, 20), 100.0) + (np.arange(5) * 0.1)[:, None, None]
    base[0, 5, 5] = 1000.0

    combined, _, _, nused = combine_flats(base, sigrej=5.0)

    assert combined[5, 5] < 150.0
    assert nused[5, 5] == 4
    np.testing.assert_allclose(combined[0, 0], 100.2, atol=0.05)


def test_build_bad_pixel_mask_flags_hot_pixel():
    flat, _, _ = make_flat(hot_pixel=(80, 300))

    mask = build_bad_pixel_mask(flat, n_sigma=5.0)

    assert mask[80, 300]
    assert mask.sum() < 200


def test_detect_order_edges_finds_21_orders():
    flat, left, right = make_flat(slope=0.0)

    detected_left, detected_right, centers = detect_order_edges(
        flat, FIRE_ORDER_SPAT_POS
    )

    assert centers.size == FIRE_NORDERS
    assert np.all(np.diff(centers) > 0)
    np.testing.assert_allclose(detected_left, left, atol=2.0)
    np.testing.assert_allclose(detected_right, right, atol=2.0)


def test_trace_orders_fits_curved_edges():
    slope = 0.05
    flat, _, _ = make_flat(slope=slope)
    nspec = flat.shape[0]

    result = trace_orders({"flat_norm": flat}, {}, {"ncoeff": 3})

    table = result.products["order_traces"]
    assert len(table) == FIRE_NORDERS
    row = table[table["order"] == 11][0]
    rows = np.array([0, nspec - 1])
    positions = evaluate_edge_trace(
        row["left_coeff"], row["xmid"], row["xscale"], rows
    )
    measured = positions[1] - positions[0]
    np.testing.assert_allclose(measured, slope * (nspec - 1), atol=1.5)
    assert result.products["edge_traces"].shape == (nspec, 2 * FIRE_NORDERS)


def test_trace_orders_single_order_retrace_preserves_other_orders():
    flat, _, _ = make_flat(slope=0.03)

    full = trace_orders({"flat_norm": flat}, {}, {"ncoeff": 3})
    baseline = full.products["order_traces"]
    before = {
        int(row["order"]): np.array(row["left_coeff"]) for row in baseline
    }

    retraced = trace_orders(
        {"flat_norm": flat, "order_traces": baseline}, {}, {"ncoeff": 3, "order": 13}
    )
    after = retraced.products["order_traces"]

    assert len(after) == FIRE_NORDERS
    target = after[after["order"] == 13][0]
    assert np.all(np.isfinite(target["left_coeff"]))
    for order, coeff in before.items():
        if order == 13:
            continue
        row = after[after["order"] == order][0]
        np.testing.assert_array_equal(np.array(row["left_coeff"]), coeff)


def test_prep_calibrations_engine_combines_and_masks():
    from onfire.core import prep_calibrations

    flat, _, _ = make_flat(hot_pixel=(80, 300))
    noisy = np.stack([flat, flat * 1.01, flat * 0.99])

    result = prep_calibrations({"raw_flats": noisy}, {}, {})

    assert result.products["flat"].shape == flat.shape
    assert result.products["bad_pixel_mask"][80, 300]
    assert "flat" in result.plots


def test_trace_orders_requires_a_flat():
    with pytest.raises(ValueError, match="flat"):
        trace_orders({}, {}, {})


def test_load_raw_frames_reads_flat_stack(tmp_path):
    flat, _, _ = make_flat()
    _write_flat(tmp_path / "f1.fits", flat)
    _write_flat(tmp_path / "f2.fits", flat * 1.02)
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})

    loaded = state.load_raw_frames(roles=["pixelflat"])

    assert loaded["raw_flats"].shape == (2, *flat.shape)
    assert state.products["raw_flats"].shape == (2, *flat.shape)


def test_state_runs_prep_and_trace_and_checkpoints(tmp_path):
    flat, _, _ = make_flat()
    _write_flat(tmp_path / "f1.fits", flat)
    _write_flat(tmp_path / "f2.fits", flat * 1.01)
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})
    state.load_raw_frames(roles=["pixelflat"])

    state.run_step("prep_calibrations")
    state.run_step("trace_orders")

    assert state.step == 2
    assert state.step_name == "trace_orders"
    assert state.products["order_traces"].colnames[0] == "order"
    assert "edge_traces" in state.plots

    path = state.save(tmp_path)
    assert path.name == "02_trace_orders.fits"
    reloaded = ReductionState.load(path)
    assert len(reloaded.products["order_traces"]) == FIRE_NORDERS
    np.testing.assert_allclose(
        reloaded.products["flat"], state.products["flat"], rtol=1e-6
    )
    original = state.products["order_traces"]
    restored = reloaded.products["order_traces"]
    for order in (11, 21, 31):
        before = original[original["order"] == order][0]
        after = restored[restored["order"] == order][0]
        np.testing.assert_array_equal(
            np.asarray(after["left_coeff"]), np.asarray(before["left_coeff"])
        )

    written = state.save_plots(tmp_path / "qa")
    assert written
    assert all(p.exists() for p in written)
    assert written[0].suffix == ".html"


def test_state_single_order_retrace_does_not_disturb_others(tmp_path):
    flat, _, _ = make_flat(slope=0.03)
    _write_flat(tmp_path / "f1.fits", flat)
    _write_flat(tmp_path / "f2.fits", flat * 1.01)
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    state = ReductionState(table, {})
    state.load_raw_frames(roles=["pixelflat"])
    state.run_step("prep_calibrations")
    state.run_step("trace_orders")
    before = {
        int(row["order"]): np.array(row["left_coeff"])
        for row in state.products["order_traces"]
    }

    state.run_step("trace_orders", order=13)

    assert state.step == 2
    after = state.products["order_traces"]
    assert len(after) == FIRE_NORDERS
    assert np.all(np.isfinite(after[after["order"] == 13][0]["left_coeff"]))
    for order, coeff in before.items():
        if order == 13:
            continue
        row = after[after["order"] == order][0]
        np.testing.assert_array_equal(np.array(row["left_coeff"]), coeff)
