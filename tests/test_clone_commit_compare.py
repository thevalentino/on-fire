"""Tests for State Clone Commit, Stale tracking, and clone comparison.

These exercise the public state seam only: ``run_step`` (with ``clone=True`` and
``order=N``), ``commit``, ``compare``, ``save`` and ``load``, driven through
registered fake Step Engines.
"""

import numpy as np
import plotly.graph_objects as go
import pytest
from astropy.table import Table

from onfire.setup import FrameTable, scan_raw_dir
from onfire.pipeline import (
    ReductionState,
    StepResult,
    clear_step_engines,
    register_step_comparison,
    register_step_engine,
)


@pytest.fixture(autouse=True)
def _isolated_step_registry():
    clear_step_engines()
    yield
    clear_step_engines()


@pytest.fixture
def make_state(make_frame, tmp_path):
    count = {"n": 0}

    def _make():
        count["n"] += 1
        make_frame(obstype="PixFlat", name=f"flat{count['n']}.fits")
        table = FrameTable.from_frames(scan_raw_dir(tmp_path))
        return ReductionState(table, {"steps": {}})

    return _make


def _engine(**products):
    """An engine returning fixed products; the ``value`` param overrides them."""

    def run(_products, _meta, params):
        out = {name: np.array([params.get("value", value)]) for name, value in products.items()}
        return StepResult(products=out)

    return run


def test_commit_replaces_baseline_in_place_and_leaves_clone_unchanged(make_state):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})

    baseline.commit(clone)

    assert baseline.products["wave"][0] == 9.0
    assert baseline.products["wave"] is not clone.products["wave"]
    clone.products["wave"][0] = -1.0
    assert baseline.products["wave"][0] == 9.0
    assert clone.step_name == "calibrate_wavelengths"


def _register_chain():
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    register_step_engine("fit_tilts", _engine(tilts=2.0))
    register_step_engine("extract_1d", _engine(spec=3.0))


def _reduced_baseline(make_state):
    baseline = make_state()
    for step in ("calibrate_wavelengths", "fit_tilts", "extract_1d"):
        baseline.run_step(step)
    return baseline


def test_commit_discards_products_made_stale_by_an_earlier_rerun(make_state):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})

    # The clone keeps its old downstream products for comparison until Commit.
    assert {"tilts", "spec"} <= set(clone.products)

    baseline.commit(clone)

    assert baseline.products["wave"][0] == 9.0
    assert "tilts" not in baseline.products
    assert "spec" not in baseline.products
    assert baseline.step_name == "calibrate_wavelengths"


def test_rerun_chain_keeps_refreshed_steps_and_drops_unrefreshed_ones(make_state):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    clone.run_step("fit_tilts", params={"value": 8.0})

    baseline.commit(clone)

    assert baseline.products["wave"][0] == 9.0
    assert baseline.products["tilts"][0] == 8.0
    assert "spec" not in baseline.products
    assert baseline.step_name == "fit_tilts"


def test_commit_without_rerun_keeps_everything(make_state):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.clone()

    baseline.commit(clone)

    assert {"wave", "tilts", "spec"} <= set(baseline.products)
    assert baseline.step_name == "extract_1d"


def test_progress_stays_contiguous_when_a_later_step_is_refreshed_over_a_stale_one(make_state):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    clone.run_step("extract_1d", params={"value": 7.0})

    baseline.commit(clone)

    assert "tilts" not in baseline.products
    assert baseline.step_name == "calibrate_wavelengths"


def _tilt_table_engine(_products, _meta, _params):
    return StepResult(products={"tilts": Table({"order": [11, 12, 13], "coeff": [1.0, 2.0, 3.0]})})


def test_single_order_rerun_discards_only_that_orders_downstream_rows(make_state):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    register_step_engine("fit_tilts", _tilt_table_engine)
    register_step_engine("coadd_orders", _engine(spliced=5.0))
    baseline = make_state()
    for step in ("calibrate_wavelengths", "fit_tilts", "coadd_orders"):
        baseline.run_step(step)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13, params={"value": 9.0})

    baseline.commit(clone)

    assert list(baseline.products["tilts"]["order"]) == [11, 12]
    assert "spliced" not in baseline.products
    assert baseline.products["wave"][0] == 9.0
    assert baseline.step_name == "calibrate_wavelengths"


def test_single_order_rerun_of_each_order_accumulates_staleness(make_state):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    register_step_engine("fit_tilts", _tilt_table_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    baseline.run_step("fit_tilts")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=11)
    clone.run_step("calibrate_wavelengths", order=13)

    baseline.commit(clone)

    assert list(baseline.products["tilts"]["order"]) == [12]


def test_full_rerun_after_single_order_rerun_clears_its_own_staleness(make_state):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    register_step_engine("fit_tilts", _tilt_table_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    baseline.run_step("fit_tilts")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13)
    clone.run_step("fit_tilts")

    baseline.commit(clone)

    assert list(baseline.products["tilts"]["order"]) == [11, 12, 13]
    assert baseline.step_name == "fit_tilts"


def test_commit_refuses_clone_after_baseline_was_run_directly(make_state):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    baseline.run_step("fit_tilts")

    with pytest.raises(ValueError, match="diverged"):
        baseline.commit(clone)

    assert baseline.products["wave"][0] == 1.0
    assert "tilts" in baseline.products


def test_commit_refuses_second_clone_after_first_was_committed(make_state):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    first = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 8.0})
    second = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    baseline.commit(first)

    with pytest.raises(ValueError, match="diverged"):
        baseline.commit(second)

    baseline.commit(second, force=True)
    assert baseline.products["wave"][0] == 9.0


def test_commit_refuses_clone_from_unrelated_state(make_state):
    _register_chain()
    one, other = make_state(), make_state()
    one.run_step("calibrate_wavelengths")
    other.run_step("calibrate_wavelengths")
    clone = other.clone()

    with pytest.raises(ValueError, match="unrelated|different state"):
        one.commit(clone)


def test_committing_a_clone_twice_requires_force(make_state):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    baseline.commit(clone)

    with pytest.raises(ValueError, match="diverged"):
        baseline.commit(clone)
    baseline.commit(clone, force=True)


def test_clone_of_a_clone_commits_into_its_immediate_parent_only(make_state):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    first = baseline.clone()
    second = first.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})

    with pytest.raises(ValueError):
        baseline.commit(second)
    first.commit(second)
    baseline.commit(first)

    assert baseline.products["wave"][0] == 9.0


def test_loaded_checkpoints_are_separate_lineages(make_state, tmp_path):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    path = baseline.save(tmp_path / "out")
    loaded_a = ReductionState.load(path)
    loaded_b = ReductionState.load(path)
    clone = loaded_a.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})

    with pytest.raises(ValueError):
        loaded_b.commit(clone)
    loaded_a.commit(clone)

    assert loaded_a.products["wave"][0] == 9.0


def _masking_engine(_products, _meta, params):
    masked = {13: params["fail"]} if params.get("fail") else {}
    return StepResult(products={"wave": np.array([1.0])}, masked_orders=masked)


def test_successful_targeted_rerun_clears_the_order_mask_and_commit_carries_it(make_state):
    register_step_engine("calibrate_wavelengths", _masking_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", params={"fail": "faint"})
    assert baseline.order_mask.is_masked(13)

    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13)

    assert not clone.order_mask.is_masked(13)
    assert baseline.order_mask.is_masked(13)
    baseline.commit(clone)
    assert not baseline.order_mask.is_masked(13)


def test_failed_targeted_rerun_keeps_the_mask_with_the_latest_reason(make_state):
    register_step_engine("calibrate_wavelengths", _masking_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", params={"fail": "faint"})

    clone = baseline.run_step(
        "calibrate_wavelengths", clone=True, order=13, params={"fail": "no lines"}
    )

    assert clone.order_mask.reasons[13] == "no lines"


def test_targeted_rerun_of_another_order_leaves_the_mask_alone(make_state):
    register_step_engine("calibrate_wavelengths", _masking_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", params={"fail": "faint"})

    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=14)

    assert clone.order_mask.is_masked(13)


def _plotting_engine(_products, _meta, params):
    value = params.get("value", 1.0)
    figure = go.Figure(go.Scatter(x=[0, 1], y=[value, value]))
    return StepResult(products={"wave": np.array([value])}, plots={"qa": figure})


def test_compare_builds_a_dual_figure_from_both_states_qa_artifacts(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 5.0})

    figure = baseline.compare(clone, "calibrate_wavelengths")

    assert len(figure.data) == 2
    assert list(figure.data[0].y) == [1.0, 1.0]
    assert list(figure.data[1].y) == [5.0, 5.0]
    assert figure.data[0].xaxis != figure.data[1].xaxis


def test_compare_accepts_a_step_ordinal(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True)

    assert len(baseline.compare(clone, 3).data) == 2


def test_compare_between_two_clones(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    first = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 2.0})
    second = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 3.0})

    figure = first.compare(second, "calibrate_wavelengths")

    assert [list(t.y)[0] for t in figure.data] == [2.0, 3.0]


def test_compare_names_the_state_missing_a_qa_artifact(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    no_qa = baseline.run_step("calibrate_wavelengths", clone=True, qa=False)

    with pytest.raises(ValueError, match="other state.*QA Artifact"):
        baseline.compare(no_qa, "calibrate_wavelengths")
    with pytest.raises(ValueError, match="this state.*QA Artifact"):
        no_qa.compare(baseline, "calibrate_wavelengths")


def test_compare_error_mentions_single_order_runs(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", qa=False)
    clone = baseline.run_step("calibrate_wavelengths", clone=True)

    with pytest.raises(ValueError, match="single-order"):
        baseline.compare(clone, "calibrate_wavelengths")


def test_a_registered_comparison_overrides_the_default(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    register_step_comparison(
        "calibrate_wavelengths",
        lambda state, other: go.Figure(layout={"title": "bespoke"}),
    )
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.run_step("calibrate_wavelengths", clone=True)

    assert baseline.compare(clone, "calibrate_wavelengths").layout.title.text == "bespoke"


def test_rerunning_a_step_without_qa_drops_its_outdated_qa_artifacts(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")

    baseline.run_step("calibrate_wavelengths", qa=False)

    assert baseline.plots == {}


def test_stale_marks_survive_a_checkpoint_round_trip(make_state, tmp_path):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))
    register_step_engine("fit_tilts", _tilt_table_engine)
    register_step_engine("extract_1d", _engine(spec=3.0))
    baseline = make_state()
    for step in ("calibrate_wavelengths", "fit_tilts", "extract_1d"):
        baseline.run_step(step)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13)

    reloaded = ReductionState.load(clone.save(tmp_path / "out"))
    reloaded.commit(reloaded.clone())

    assert list(reloaded.products["tilts"]["order"]) == [11, 12]
    assert "spec" not in reloaded.products
    assert reloaded.step_name == "calibrate_wavelengths"


def test_checkpoint_saved_after_commit_holds_no_stale_products(make_state, tmp_path):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    baseline.commit(clone)

    reloaded = ReductionState.load(baseline.save(tmp_path / "out"))
    reloaded.commit(reloaded.clone())

    assert set(reloaded.products) == {"wave"}
    assert reloaded.step_name == "calibrate_wavelengths"


def test_commit_writes_nothing_to_disk(make_state, tmp_path):
    _register_chain()
    baseline = _reduced_baseline(make_state)
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})
    before = sorted(p.name for p in tmp_path.iterdir())

    baseline.commit(clone)

    assert sorted(p.name for p in tmp_path.iterdir()) == before


def _telluric_state(make_state):
    state = make_state()
    for method, level in (("empirical", 0.5), ("grid", 0.8)):
        for suffix in ("transmission", "flux", "var", "ivar", "mask"):
            state.products[f"telluric_{method}_{suffix}"] = np.full((4, 1), level)
    state.products["telluric_wave"] = np.arange(4.0).reshape(4, 1)
    return state


def test_compare_telluric_select_picks_the_method(make_state):
    state = _telluric_state(make_state)

    state.compare_telluric(select="grid")

    assert state.products["telluric_flux"][0, 0] == 0.8
    assert state.meta["telluric_method"] == "grid"


def test_compare_telluric_commit_argument_is_deprecated_but_still_works(make_state):
    state = _telluric_state(make_state)

    with pytest.warns(DeprecationWarning, match="select"):
        state.compare_telluric(commit="empirical")

    assert state.meta["telluric_method"] == "empirical"


def test_commit_drops_qa_artifacts_of_discarded_steps(make_state):
    register_step_engine("calibrate_wavelengths", _plotting_engine)

    def tilt_engine(_p, _m, _q):
        return StepResult(
            products={"tilts": np.array([2.0])}, plots={"tilt_qa": go.Figure()}
        )

    register_step_engine("fit_tilts", tilt_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    baseline.run_step("fit_tilts")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, params={"value": 9.0})

    baseline.commit(clone)

    assert set(baseline.plots) == {"qa"}


def _per_order_plot_engine(_products, _meta, params):
    order = params.get("order", 11)
    return StepResult(
        products={"wave": np.array([1.0])},
        plots={f"qa_{order}": go.Figure(go.Scatter(x=[0, 1], y=[order, order]))},
    )


def test_single_order_rerun_keeps_the_other_orders_qa_artifacts(make_state):
    register_step_engine("calibrate_wavelengths", _per_order_plot_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", order=11)
    baseline.run_step("calibrate_wavelengths", order=12)

    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13)

    assert set(clone.plots) == {"qa_11", "qa_12", "qa_13"}
    assert set(baseline.plots) == {"qa_11", "qa_12"}


def test_single_order_rerun_on_the_baseline_keeps_other_orders_qa_artifacts(make_state):
    register_step_engine("calibrate_wavelengths", _per_order_plot_engine)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths", order=11)
    baseline.run_step("calibrate_wavelengths", order=12)

    assert set(baseline.plots) == {"qa_11", "qa_12"}


def test_cross_order_step_is_discarded_whole_when_any_input_order_is_stale(make_state):
    register_step_engine("calibrate_wavelengths", _engine(wave=1.0))

    def coadd(_p, _m, _q):
        return StepResult(
            products={"spliced": Table({"order": [11, 12, 13], "flux": [1.0, 2.0, 3.0]})}
        )

    register_step_engine("coadd_orders", coadd)
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    baseline.run_step("coadd_orders")
    clone = baseline.run_step("calibrate_wavelengths", clone=True, order=13)

    baseline.commit(clone)

    assert "spliced" not in baseline.products
    assert baseline.step_name == "calibrate_wavelengths"


def test_commit_carries_the_clones_parameter_config(make_state):
    _register_chain()
    baseline = make_state()
    baseline.run_step("calibrate_wavelengths")
    clone = baseline.clone()
    clone.config["steps"]["calibrate_wavelengths"] = {"value": 4.0}
    clone.run_step("calibrate_wavelengths")

    baseline.commit(clone)

    assert baseline.config["steps"]["calibrate_wavelengths"] == {"value": 4.0}
    assert baseline.products["wave"][0] == 4.0
    assert baseline.config is not clone.config
