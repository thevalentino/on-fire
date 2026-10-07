"""High-level tests for the ReductionState container.

These exercise the public state seam: construction from a Frame Table and YAML
Parameter Config, step execution with cloning, Multi-Extension FITS checkpoint
round-tripping, and Plotly QA artifact handling.
"""

import numpy as np
import pytest
from astropy.table import Table

from onfire.setup import FrameTable, scan_raw_dir
from onfire.pipeline import (
    DEFAULT_CONFIG,
    ReductionState,
    StepResult,
    clear_step_engines,
    load_config,
    register_step_engine,
)


@pytest.fixture(autouse=True)
def _isolated_step_registry():
    clear_step_engines()
    yield
    clear_step_engines()


def _frame_table(make_frame, tmp_path):
    make_frame(obstype="PixFlat", name="a.fits")
    make_frame(obstype="Science", name="b.fits")
    return FrameTable.from_frames(scan_raw_dir(tmp_path))


def test_state_initializes_from_frame_table_and_yaml_config(make_frame, tmp_path):
    table = _frame_table(make_frame, tmp_path)
    frames_path = tmp_path / "frames.ascii"
    table.write(frames_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: stare\nsteps:\n  trace_orders:\n    ncoeff: 5\n")

    state = ReductionState.from_files(frames_path, config_path)

    assert [f.filename for f in state.frames.frames] == ["a.fits", "b.fits"]
    assert state.config["mode"] == "stare"
    assert state.config["steps"]["trace_orders"]["ncoeff"] == 5
    assert state.config["qa_dir"] == "qa"
    assert state.step == 0
    assert state.products == {}
    assert state.plots == {}


def test_state_constructor_accepts_objects_directly(make_frame, tmp_path):
    table = _frame_table(make_frame, tmp_path)

    state = ReductionState(table, {"mode": "pair_difference"})

    assert state.step_name == ""
    assert state.frames is table
    assert state.config["mode"] == "pair_difference"


def _state(make_frame, tmp_path):
    return ReductionState(_frame_table(make_frame, tmp_path), {"steps": {}})


def test_run_step_mutates_state_in_place(make_frame, tmp_path):
    seen = {}

    def engine(products, meta, params):
        seen["params"] = params
        return StepResult(
            products={"detector": np.ones((3, 3))},
            meta={"n_orders": 21},
            masked_orders={13: "faint"},
        )

    register_step_engine("trace_orders", engine)
    state = _state(make_frame, tmp_path)

    returned = state.run_step("trace_orders", params={"ncoeff": 5})

    assert returned is state
    assert state.step == 2
    assert state.step_name == "trace_orders"
    assert state.products["detector"].shape == (3, 3)
    assert state.meta["n_orders"] == 21
    assert state.order_mask.is_masked(13)
    assert state.order_mask.reasons[13] == "faint"
    assert seen["params"] == {"ncoeff": 5}


def test_run_step_accepts_ordinal(make_frame, tmp_path):
    register_step_engine("trace_orders", lambda p, m, q: StepResult(meta={"ok": True}))
    state = _state(make_frame, tmp_path)

    state.run_step(2)

    assert state.step_name == "trace_orders"
    assert state.meta["ok"] is True


def test_run_step_forwards_single_order(make_frame, tmp_path):
    seen = {}

    def engine(products, meta, params):
        seen.update(params)
        return StepResult()

    register_step_engine("trace_orders", engine)
    state = _state(make_frame, tmp_path)

    state.run_step("trace_orders", order=13)

    assert seen["order"] == 13


def test_run_step_clone_returns_independent_state(make_frame, tmp_path):
    def mutating_engine(products, meta, params):
        products["touched"] = np.array([1.0, 2.0])
        meta["ran"] = True
        return StepResult(products={"extra": np.arange(3)})

    register_step_engine("trace_orders", mutating_engine)
    baseline = _state(make_frame, tmp_path)
    baseline.products["original"] = np.zeros((2, 2))

    clone = baseline.run_step("trace_orders", clone=True)

    assert clone is not baseline
    assert baseline.step == 0
    assert "touched" not in baseline.products
    assert "extra" not in baseline.products
    assert "ran" not in baseline.meta
    assert clone.step == 2
    assert set(clone.products) == {"original", "touched", "extra"}
    assert clone.products["original"] is not baseline.products["original"]


def test_run_step_unknown_engine_reports_clear_error(make_frame, tmp_path):
    state = _state(make_frame, tmp_path)

    with pytest.raises(KeyError, match="no engine registered"):
        state.run_step("extract_1d")


def test_checkpoint_filename_uses_chronological_prefix(make_frame, tmp_path):
    state = _state(make_frame, tmp_path)
    assert state.checkpoint_filename() == "00_initial.fits"

    register_step_engine("trace_orders", lambda p, m, q: StepResult())
    state.run_step("trace_orders")

    assert state.checkpoint_filename() == "02_trace_orders.fits"


def test_save_and_load_roundtrips_without_information_loss(make_frame, tmp_path):
    register_step_engine(
        "trace_orders",
        lambda p, m, q: StepResult(meta={"n_orders": 21}),
    )
    state = _state(make_frame, tmp_path)
    state.source = str(tmp_path)
    state.run_step("trace_orders")
    state.products["flux"] = np.arange(12, dtype=np.float32).reshape(3, 4)
    state.products["bad_mask"] = np.array([[True, False], [False, True]])
    state.products["traces"] = Table({"order": [11, 12], "xcen": [1.5, 2.5]})
    state.meta["rms"] = 0.5
    state.meta["labels"] = ["a", "b"]
    state.order_mask.mask(13, "faint")

    path = state.save(tmp_path)
    reloaded = ReductionState.load(path)

    assert path.name == "02_trace_orders.fits"
    assert reloaded.step == 2
    assert reloaded.step_name == "trace_orders"
    assert reloaded.source == state.source
    assert reloaded.frames.source == state.frames.source
    assert reloaded.meta == state.meta
    assert reloaded.order_mask.reasons == {13: "faint"}
    assert [f.filename for f in reloaded.frames.frames] == ["a.fits", "b.fits"]


def test_checkpoint_preserves_array_dtypes_and_shapes(make_frame, tmp_path):
    state = _state(make_frame, tmp_path)
    state.products["float64"] = np.linspace(0, 1, 5)
    state.products["float32"] = np.arange(6, dtype=np.float32)
    state.products["int16"] = np.array([1, 2, 3], dtype=np.int16)
    state.products["bool_mask"] = np.array([True, False, True])

    path = state.save(tmp_path)
    reloaded = ReductionState.load(path)

    for key, original in state.products.items():
        restored = reloaded.products[key]
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape
        np.testing.assert_array_equal(restored, original)


def test_checkpoint_preserves_astropy_table_columns(make_frame, tmp_path):
    state = _state(make_frame, tmp_path)
    state.products["objs"] = Table(
        {"order": np.array([11, 12], dtype=np.int16), "snr": [3.5, 4.5]},
        names=("order", "snr"),
    )

    path = state.save(tmp_path)
    reloaded = ReductionState.load(path)
    restored = reloaded.products["objs"]

    assert restored.colnames == ["order", "snr"]
    assert list(restored["order"]) == [11, 12]
    assert list(restored["snr"]) == [3.5, 4.5]


def test_checkpoint_into_missing_directory_is_created(make_frame, tmp_path):
    state = _state(make_frame, tmp_path)
    dest = tmp_path / "checkpoints"

    path = state.save(dest)

    assert path.parent == dest
    assert path.exists()


def _plotting_engine(p, m, q):
    import plotly.graph_objects as go

    return StepResult(plots={"edge_traces": go.Figure()})


def test_run_step_attaches_plotly_figures_by_default(make_frame, tmp_path):
    register_step_engine("trace_orders", _plotting_engine)
    state = _state(make_frame, tmp_path)

    state.run_step("trace_orders")

    assert "edge_traces" in state.plots


def test_run_step_qa_false_skips_plot_attachment(make_frame, tmp_path):
    register_step_engine("trace_orders", _plotting_engine)
    state = _state(make_frame, tmp_path)

    state.run_step("trace_orders", qa=False)

    assert state.plots == {}


def test_add_plot_attaches_figure(make_frame, tmp_path):
    import plotly.graph_objects as go

    state = _state(make_frame, tmp_path)
    figure = go.Figure()

    state.add_plot("custom", figure)

    assert state.plots["custom"] is figure


def test_save_plots_writes_standalone_html(make_frame, tmp_path):
    register_step_engine("trace_orders", _plotting_engine)
    state = _state(make_frame, tmp_path)
    state.run_step("trace_orders")
    qa_dir = tmp_path / "qa"

    written = state.save_plots(qa_dir)

    assert [p.name for p in written] == ["02_trace_orders_edge_traces.html"]
    assert written[0].exists()
    assert "<html" in written[0].read_text().lower()


def test_default_config_nested_values_are_not_aliased(make_frame, tmp_path):
    table = _frame_table(make_frame, tmp_path)
    first = ReductionState(table, load_config(None))
    second = ReductionState(table, load_config(None))

    first.config["steps"]["trace_orders"] = {"ncoeff": 5}

    assert "trace_orders" not in second.config["steps"]
    assert DEFAULT_CONFIG["steps"] == {}


def test_clone_deep_copies_plots(make_frame, tmp_path):
    import plotly.graph_objects as go

    baseline = _state(make_frame, tmp_path)
    baseline.add_plot("qa", go.Figure())

    clone = baseline.clone()

    assert clone.plots["qa"] is not baseline.plots["qa"]


def test_rerunning_earlier_step_preserves_progress(make_frame, tmp_path):
    register_step_engine("trace_orders", lambda p, m, q: StepResult())
    register_step_engine("prep_calibrations", lambda p, m, q: StepResult())
    state = _state(make_frame, tmp_path)

    state.run_step("trace_orders")
    state.run_step("prep_calibrations", order=13)

    assert state.step == 2
    assert state.checkpoint_filename() == "02_trace_orders.fits"


def test_checkpoint_preserves_frame_paths(make_frame, tmp_path):
    table = _frame_table(make_frame, tmp_path)
    state = ReductionState(table, {})
    original_paths = [f.path for f in state.frames.frames]

    reloaded = ReductionState.load(state.save(tmp_path))

    assert [f.path for f in reloaded.frames.frames] == original_paths


def test_run_step_rejects_non_step_result(make_frame, tmp_path):
    register_step_engine("trace_orders", lambda p, m, q: "not a result")
    state = _state(make_frame, tmp_path)

    with pytest.raises(TypeError, match="expected StepResult"):
        state.run_step("trace_orders")
