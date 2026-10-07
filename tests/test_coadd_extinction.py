"""High-level tests for order coaddition and Galactic dust extinction (steps 11-12).

These exercise the public seams for the final two reduction steps: the pure
``onfire.core.coadd`` and ``onfire.core.extinction`` functions, the
``coadd_orders`` and ``extinction_correct`` Step Engines, and the
``ReductionState`` API including Multi-Extension FITS persistence.
"""

import numpy as np
import pytest
from astropy.table import Table

from onfire.core import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    coadd_order_spectra,
    coadd_orders,
    deredden_spectrum,
    extinction_correct,
    extinction_factor,
    fitzpatrick99,
    query_ebv,
    register_engines,
)
from onfire.pipeline import ReductionState, clear_step_engines
from onfire.setup import FrameTable


@pytest.fixture(autouse=True)
def _engines():
    clear_step_engines()
    register_engines()
    yield
    clear_step_engines()


def _smooth(wave, scale=1.0e-13, index=-2.0):
    return scale * (wave / 1.5e4) ** index


# --------------------------------------------------------------------------- #
# Pure Galactic extinction functions
# --------------------------------------------------------------------------- #


def test_fitzpatrick99_anchors_to_rv_at_v_band():
    # A(V) = R_V * E(B-V) is the definition of the total-to-selective ratio.
    for rv in (3.1, 4.0, 5.0):
        a_v = fitzpatrick99(np.array([1.0e4 / 1.82]), ebv=1.0, rv=rv)[0]
        assert a_v == pytest.approx(rv, rel=1e-6)


def test_fitzpatrick99_infrared_power_law():
    # In the infrared the F99 curve is A(lambda) ~ lambda**-1.61.
    wave = np.array([10000.0, 11000.0])
    a = fitzpatrick99(wave, ebv=1.0, rv=3.1)
    assert a[0] / a[1] == pytest.approx((1.1) ** 1.61, rel=1e-6)


def test_fitzpatrick99_is_continuous_at_branch_edges():
    for edge in (1.1, 3.3):
        wave = 1.0e4 / edge
        eps = 1.0e-4
        below = fitzpatrick99(np.array([1.0e4 / (edge - eps)]), ebv=1.0)[0]
        above = fitzpatrick99(np.array([1.0e4 / (edge + eps)]), ebv=1.0)[0]
        assert below == pytest.approx(above, rel=1e-3)


def test_fitzpatrick99_scales_linearly_with_ebv():
    wave = np.linspace(1.0e4, 2.4e4, 50)
    unit = fitzpatrick99(wave, ebv=1.0)
    double = fitzpatrick99(wave, ebv=0.5)
    np.testing.assert_allclose(double, unit / 2.0)


def test_extinction_factor_is_ten_to_the_point_four_a():
    wave = np.linspace(1.0e4, 2.4e4, 50)
    a = fitzpatrick99(wave, ebv=0.4)
    factor = extinction_factor(wave, ebv=0.4)
    np.testing.assert_allclose(factor, 10.0 ** (0.4 * a))


def test_deredden_spectrum_recovers_intrinsic_flux_and_variance():
    wave = np.linspace(1.0e4, 2.4e4, 300)
    intrinsic = _smooth(wave)
    ebv = 0.35
    a = fitzpatrick99(wave, ebv=ebv)
    observed = intrinsic * 10.0 ** (-0.4 * a)
    variance = 0.01 * observed**2

    flux, out_var, out_ivar, mask = deredden_spectrum(
        wave, observed, ebv, variance=variance
    )

    assert mask.all()
    np.testing.assert_allclose(flux, intrinsic, rtol=1e-9)
    expected_var = variance * (10.0 ** (0.4 * a)) ** 2
    np.testing.assert_allclose(out_var, expected_var, rtol=1e-9)
    np.testing.assert_allclose(out_ivar, 1.0 / expected_var, rtol=1e-9)


def test_query_ebv_interpolates_local_map():
    catalog = Table(
        {
            "ra": np.array([10.0, 20.0, 30.0]),
            "dec": np.array([0.0, 0.0, 0.0]),
            "ebv": np.array([0.1, 0.2, 0.3]),
        }
    )

    assert query_ebv(20.0, 0.0, catalog=catalog) == pytest.approx(0.2)
    assert query_ebv(19.0, 0.5, catalog=catalog) == pytest.approx(0.2)


def test_query_ebv_without_map_is_an_error():
    with pytest.raises(ValueError, match="no dust map"):
        query_ebv(10.0, 0.0)


# --------------------------------------------------------------------------- #
# Pure coaddition functions
# --------------------------------------------------------------------------- #


def _segments():
    """Return three overlapping band-limited segments of one smooth spectrum."""
    wave_a = np.linspace(1.0e4, 1.6e4, 200)
    wave_b = np.linspace(1.4e4, 2.0e4, 200)
    wave_c = np.linspace(1.8e4, 2.4e4, 200)
    segments = []
    for wave in (wave_a, wave_b, wave_c):
        segments.append((wave, _smooth(wave), np.full(wave.size, 2.0)))
    return segments


def test_coadd_order_spectra_spans_full_wavelength_range():
    segments = _segments()
    coadd = coadd_order_spectra(
        [s[0] for s in segments], [s[1] for s in segments], [s[2] for s in segments]
    )

    assert coadd.wave.min() <= 1.01e4
    assert coadd.wave.max() >= 2.39e4
    # The coadded spectrum tracks the injected smooth continuum.
    expected = _smooth(coadd.wave)
    np.testing.assert_allclose(coadd.flux[coadd.mask], expected[coadd.mask], rtol=1e-2)
    assert coadd.norders == 3


def test_coadd_order_spectra_overlap_is_inverse_variance_weighted():
    wave = np.linspace(1.0e4, 1.2e4, 100)
    good = np.full(wave.size, 1.0)
    noisy = np.full(wave.size, 4.0)
    f_good = np.full(wave.size, 10.0)
    f_noisy = np.full(wave.size, 0.0)

    coadd = coadd_order_spectra([wave, wave], [f_good, f_noisy], [good, noisy])

    # With weights 1.0 and 0.25 the weighted mean is (10*1 + 0*0.25)/1.25 = 8.
    valid = coadd.mask
    np.testing.assert_allclose(coadd.flux[valid], 8.0)
    np.testing.assert_allclose(coadd.var[valid], 1.0 / 1.25)


def test_coadd_order_spectra_skips_all_nan_segments():
    wave = np.linspace(1.0e4, 1.2e4, 50)
    flux = np.full(50, np.nan)
    empty = coadd_order_spectra([wave], [flux], [np.ones(50)])

    assert not empty.mask.any()
    assert empty.flux.size == 0


# --------------------------------------------------------------------------- #
# coadd_orders and extinction_correct Step Engines
# --------------------------------------------------------------------------- #


def _coadd_products():
    """One bright object in every order, flux 1e-13, flat variance."""
    wave = np.linspace(1.0e4, 2.4e4, 200)
    wave_soln = np.tile(wave[:, None], (1, FIRE_NORDERS))
    orders = FIRE_ORDER_NUMBERS.copy()
    objects = Table(
        {
            "order": orders.astype(np.int16),
            "obj_id": np.zeros(FIRE_NORDERS, dtype=np.int32),
            "snr": np.full(FIRE_NORDERS, 20.0, dtype=np.float32),
        }
    )
    flux = np.tile(_smooth(wave)[:, None], (1, FIRE_NORDERS))
    var = 0.01 * flux**2
    mask = np.ones(flux.shape, dtype=bool)
    return {
        "wave_soln": wave_soln,
        "objects": objects,
        "flux": flux,
        "flux_var": var,
        "flux_mask": mask,
    }


def test_coadd_orders_engine_builds_continuous_spectrum():
    products = _coadd_products()

    result = coadd_orders(products, {}, {})

    assert result.products["coadd_flux"].ndim == 1
    assert result.products["coadd_wave"].ndim == 1
    assert result.meta["n_coadd_orders"] == FIRE_NORDERS
    valid = result.products["coadd_mask"]
    assert valid.any()
    np.testing.assert_allclose(
        result.products["coadd_flux"][valid],
        _smooth(result.products["coadd_wave"])[valid],
        rtol=2e-2,
    )
    assert "coadd_orders" in result.plots


def test_coadd_orders_requires_flux_calibrated_input():
    with pytest.raises(ValueError, match="coadd_orders requires"):
        coadd_orders({}, {}, {})


def test_coadd_orders_prefers_telluric_corrected_flux():
    products = _coadd_products()
    products["telluric_flux"] = products["flux"] * 0.5
    products["telluric_var"] = products["flux_var"] * 0.25
    products["telluric_mask"] = products["flux_mask"]

    result = coadd_orders(products, {}, {})

    valid = result.products["coadd_mask"]
    np.testing.assert_allclose(
        result.products["coadd_flux"][valid],
        0.5 * _smooth(result.products["coadd_wave"])[valid],
        rtol=2e-2,
    )


def _extinction_products():
    products = _coadd_products()
    coadd = coadd_orders(products, {}, {})
    products.update(coadd.products)
    return products


def test_extinction_correct_engine_uses_config_override():
    products = _extinction_products()

    result = extinction_correct(products, {"ra": 10.0, "dec": 0.0}, {"ebv": 0.25})

    assert result.meta["ebv"] == pytest.approx(0.25)
    assert result.meta["rv"] == pytest.approx(3.1)
    assert "final_flux" in result.products
    assert "extinction_correct" in result.plots


def test_extinction_correct_requires_coadded_spectrum():
    with pytest.raises(ValueError, match="extinction_correct requires"):
        extinction_correct({}, {}, {})


def test_extinction_correct_queries_dust_map_from_coordinates():
    products = _extinction_products()
    products["dust_map"] = Table(
        {"ra": [10.0], "dec": [0.0], "ebv": [0.4]}
    )

    result = extinction_correct(products, {"ra": 10.0, "dec": 0.0}, {})

    assert result.meta["ebv"] == pytest.approx(0.4)


def test_extinction_correct_needs_ebv_or_coordinates():
    products = _extinction_products()

    with pytest.raises(ValueError, match="needs an 'ebv'"):
        extinction_correct(products, {}, {})


# --------------------------------------------------------------------------- #
# ReductionState persistence of the final spectrum
# --------------------------------------------------------------------------- #


def test_state_coadd_and_extinction_checkpoint_round_trip(tmp_path):
    state = ReductionState(FrameTable.from_frames([]), {})
    state.products.update(_coadd_products())
    state.step = 10

    state.run_step("coadd_orders")
    assert state.step == 11
    assert state.checkpoint_filename() == "11_coadd_orders.fits"

    state.run_step("extinction_correct", params={"ebv": 0.2})
    assert state.step == 12
    assert state.checkpoint_filename() == "12_extinction_correct.fits"

    path = state.save(tmp_path)
    reloaded = ReductionState.load(path)

    for key in ("coadd_flux", "final_flux", "final_ivar"):
        np.testing.assert_allclose(
            reloaded.products[key], state.products[key], equal_nan=True
        )
    assert reloaded.meta["ebv"] == pytest.approx(0.2)
