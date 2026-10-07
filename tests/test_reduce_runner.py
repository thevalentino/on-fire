"""High-level tests for the end-to-end orchestration runner and CLI.

These exercise the public runner seam (:func:`onfire.reduce`), the reduction
step sequence controls (``--until``, ``--from``), default batch order masking,
and ``--fail-fast`` reproduction reporting, plus the ``onfire reduce`` command.
"""

import numpy as np
import pytest

from onfire.cli import build_parser, main
from onfire.pipeline import (
    FailFastError,
    STEP_SEQUENCE,
    StepResult,
    clear_step_engines,
    register_step_engine,
)
from onfire.pipeline.runner import format_reproduction_code, reduce
from onfire.setup import FrameTable, scan_raw_dir

import onfire


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_step_engines()
    yield
    clear_step_engines()


def _frames_file(make_frame, tmp_path):
    make_frame(obstype="PixFlat", name="flat.fits")
    make_frame(obstype="Arc", name="arc.fits")
    make_frame(obstype="Science", name="sci.fits")
    make_frame(obstype="Telluric", name="std.fits")
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    path = tmp_path / "frames.ascii"
    table.write(path)
    return path


def _register_stubs(counter=None, mask_step=None, raise_step=None):
    """Register a trivial engine for every step, recording invocations."""
    clear_step_engines()

    def engine_for(name):
        def engine(products, meta, params):
            if counter is not None:
                counter[name] = counter.get(name, 0) + 1
            if raise_step == name:
                raise RuntimeError(f"{name} blew up")
            masked = {13: "faint"} if mask_step == name else {}
            return StepResult(
                products={f"product_{name}": np.ones(3)},
                meta={f"meta_{name}": True},
                masked_orders=masked,
            )

        return engine

    for name in STEP_SEQUENCE:
        register_step_engine(name, engine_for(name))


def test_reduce_runs_full_sequence_and_saves_checkpoints(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    counter: dict[str, int] = {}
    _register_stubs(counter)
    out = tmp_path / "ckpt"

    state = reduce(frames, output_dir=out, qa=False, load_raw=False, log=None)

    assert state.step == len(STEP_SEQUENCE)
    assert state.step_name == "extinction_correct"
    assert all(counter[name] == 1 for name in STEP_SEQUENCE)
    assert (out / "12_extinction_correct.fits").exists()
    assert state.products["product_coadd_orders"].shape == (3,)


def test_reduce_until_stops_after_requested_step(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    counter: dict[str, int] = {}
    _register_stubs(counter)

    state = reduce(
        frames, until="trace_orders", output_dir=tmp_path / "ckpt",
        qa=False, load_raw=False, log=None,
    )

    assert state.step == 2
    assert counter == {"prep_calibrations": 1, "trace_orders": 1}


def test_reduce_from_resumes_checkpoint_without_rerunning(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    out = tmp_path / "ckpt"
    _register_stubs()
    reduce(frames, until="trace_orders", output_dir=out, qa=False, load_raw=False, log=None)

    counter: dict[str, int] = {}
    _register_stubs(counter)
    state = reduce(
        frames, from_step="calibrate_wavelengths", output_dir=out,
        qa=False, load_raw=False, log=None,
    )

    assert "prep_calibrations" not in counter
    assert "trace_orders" not in counter
    assert counter["calibrate_wavelengths"] == 1
    assert state.step == len(STEP_SEQUENCE)


def test_reduce_from_missing_checkpoint_is_an_error(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    _register_stubs()

    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        reduce(
            frames, from_step="extract_1d", output_dir=tmp_path / "empty",
            qa=False, load_raw=False, log=None,
        )


def test_reduce_batch_mode_masks_failed_order_and_continues(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    counter: dict[str, int] = {}
    _register_stubs(counter, mask_step="trace_orders")

    state = reduce(
        frames, output_dir=tmp_path / "ckpt", qa=False, load_raw=False, log=None
    )

    assert state.step == len(STEP_SEQUENCE)
    assert state.order_mask.is_masked(13)
    assert state.order_mask.reasons[13] == "faint"
    assert counter["extract_1d"] == 1


def test_reduce_fail_fast_halts_on_masked_order(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    counter: dict[str, int] = {}
    _register_stubs(counter, mask_step="trace_orders")

    with pytest.raises(FailFastError) as excinfo:
        reduce(
            frames, output_dir=tmp_path / "ckpt", fail_fast=True,
            qa=False, load_raw=False, log=None,
        )

    error = excinfo.value
    assert error.step == "trace_orders"
    assert error.order == 13
    assert "trace_orders" not in counter or counter["trace_orders"] == 1
    assert "calibrate_wavelengths" not in counter
    assert "ReductionState.load" in error.reproduction_code
    assert "'trace_orders'" in error.reproduction_code
    assert "order=13" in error.reproduction_code
    # Reproduction reloads the pre-failure checkpoint, not the post-step state.
    assert "01_prep_calibrations.fits" in error.reproduction_code


def test_reduce_fail_fast_reports_code_for_exception(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("mode: stare\n")

    _register_stubs(raise_step="extract_1d")
    with pytest.raises(FailFastError) as excinfo:
        reduce(
            frames, config, output_dir=tmp_path / "ckpt", fail_fast=True,
            qa=False, load_raw=False, log=None,
        )

    code = excinfo.value.reproduction_code
    assert "'extract_1d'" in code
    assert "'mode': 'stare'" in code
    assert "onfire.ReductionState.load" in code
    # The failed step did not advance the state, so the prior checkpoint survives.
    assert "07_subtract_sky.fits" in code


def test_reduce_without_fail_fast_propagates_step_exception(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    _register_stubs(raise_step="extract_1d")

    with pytest.raises(RuntimeError, match="extract_1d blew up"):
        reduce(
            frames, output_dir=tmp_path / "ckpt", fail_fast=False,
            qa=False, load_raw=False, log=None,
        )


def test_format_reproduction_code_is_copy_pasteable():
    code = format_reproduction_code(
        "fit_tilts", {"mode": "stare", "spec_order": 3}, checkpoint="07_x.fits", order=13
    )

    compile(code, "<reproduction>", "exec")
    assert code.splitlines()[0] == "state = onfire.ReductionState.load('07_x.fits')"
    assert "order=13" in code
    assert "'spec_order': 3" in code


def test_reduce_attaches_fits_target_coordinates(make_frame, tmp_path):
    make_frame(
        obstype="Science", name="sci.fits", extra={"RA": 123.4, "DEC": -5.6}
    )
    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    frames = tmp_path / "frames.ascii"
    table.write(frames)
    seen: dict = {}

    def extinction_engine(products, meta, params):
        seen.update(meta)
        return StepResult()

    _register_stubs()
    register_step_engine("extinction_correct", extinction_engine)

    reduce(frames, output_dir=tmp_path / "ckpt", qa=False, load_raw=True, log=None)

    assert seen.get("ra") == pytest.approx(123.4)
    assert seen.get("dec") == pytest.approx(-5.6)


def test_top_level_onfire_exposes_runner():
    assert onfire.reduce is reduce
    assert onfire.run is reduce


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_reduce_runs_and_writes_checkpoints(make_frame, tmp_path):
    frames = _frames_file(make_frame, tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("mode: stare\n")
    _register_stubs()
    out = tmp_path / "ckpt"

    rc = main(
        [
            "reduce",
            str(config),
            "--frames",
            str(frames),
            "--until",
            "trace_orders",
            "--output-dir",
            str(out),
            "--no-qa",
        ]
    )

    assert rc == 0
    assert (out / "02_trace_orders.fits").exists()


def test_cli_reduce_fail_fast_prints_reproduction_code(make_frame, tmp_path, capsys):
    frames = _frames_file(make_frame, tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("mode: stare\n")
    _register_stubs(raise_step="extract_1d")

    rc = main(
        [
            "reduce",
            str(config),
            "--frames",
            str(frames),
            "--fail-fast",
            "--output-dir",
            str(tmp_path / "ckpt"),
            "--no-qa",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "onfire.ReductionState.load" in err
    assert "'extract_1d'" in err


def test_cli_reduce_reports_missing_frames(make_frame, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("mode: stare\n")

    rc = main(["reduce", str(config), "--frames", str(tmp_path / "nope.ascii")])

    assert rc == 2
    assert "error" in capsys.readouterr().err.lower()


def test_cli_reduce_accepts_interactive_flag():
    args = build_parser().parse_args(
        ["reduce", "config.yaml", "--until", "coadd_orders", "--interactive"]
    )

    assert args.interactive is True
    assert args.until == "coadd_orders"
