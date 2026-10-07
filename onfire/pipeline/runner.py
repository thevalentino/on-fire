"""End-to-end orchestration runner for the onfire reduction pipeline.

The runner drives the canonical 12-step sequence over a
:class:`~onfire.pipeline.state.ReductionState`, loading the raw exposures each
step needs, saving Multi-Extension FITS checkpoints and Plotly QA artifacts, and
honouring the ``--until``, ``--from`` and ``--fail-fast`` controls exposed on the
``onfire reduce`` command line.
"""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any, Callable, Optional

from onfire.pipeline.config import load_config
from onfire.pipeline.raw import read_target_coordinates
from onfire.pipeline.state import ReductionState
from onfire.pipeline.steps import STEP_SEQUENCE, step_name, step_ordinal

# Frame Table roles that must be loaded from disk before each step runs.
STEP_ROLES: dict[str, tuple[str, ...]] = {
    "prep_calibrations": ("pixelflat",),
    "trace_orders": ("pixelflat",),
    "calibrate_wavelengths": ("arc",),
    "fit_tilts": ("arc",),
    "process_science": ("science",),
    "find_objects": (),
    "subtract_sky": (),
    "extract_1d": (),
    "flux_calibrate": ("standard",),
    "telluric_correct": ("standard",),
    "coadd_orders": (),
    "extinction_correct": (),
}


class FailFastError(RuntimeError):
    """Raised when ``--fail-fast`` halts a reduction at a failing step.

    Attributes:
        step: The reduction step that failed.
        state: The state as it was when execution halted.
        params: The effective parameters the step was called with.
        order: The single order that failed, if the failure was order-scoped.
        checkpoint: The checkpoint that reproduces the pre-failure state.
        reproduction_code: Copy-pasteable Python that reloads the state and
            re-runs the failed step with identical arguments.
    """

    def __init__(
        self,
        message: str,
        *,
        step: str,
        state: ReductionState,
        params: dict[str, Any],
        order: Optional[int],
        checkpoint: Optional[Path],
        reproduction_code: str,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.state = state
        self.params = params
        self.order = order
        self.checkpoint = checkpoint
        self.reproduction_code = reproduction_code


def _resolve_ordinal(value: int | str | None) -> Optional[int]:
    """Resolve a step name or ordinal to a 1-based ordinal, or ``None``."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    return step_ordinal(text)


def format_reproduction_code(
    step: str,
    params: dict[str, Any],
    *,
    checkpoint: Optional[str | Path] = None,
    order: Optional[int] = None,
    qa: bool = True,
) -> str:
    """Return copy-pasteable Python that reloads a state and re-runs a step.

    Args:
        step: Reduction step name to re-run.
        params: Effective parameters to pass unchanged.
        checkpoint: Checkpoint path to load before re-running.
        order: Optional single-order target.
        qa: Whether to attach QA figures when re-running.

    Returns:
        A short Python snippet suitable for an IPython session.
    """
    location = str(checkpoint) if checkpoint is not None else "<checkpoint path>"
    params_repr = pprint.pformat(dict(params), sort_dicts=True, width=88)
    return (
        f"state = onfire.ReductionState.load({location!r})\n"
        f"state = state.run_step({step!r}, params={params_repr}, "
        f"order={order!r}, qa={qa!r})\n"
    )


def _checkpoint_path(directory: Path, ordinal: int) -> Path:
    """Return the checkpoint path for a completed step ordinal."""
    if ordinal <= 0:
        return directory / "00_initial.fits"
    return directory / f"{ordinal:02d}_{step_name(ordinal)}.fits"


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def reduce(
    frames: str | Path | None = None,
    config: str | Path | None = None,
    *,
    until: int | str | None = None,
    from_step: int | str | None = None,
    fail_fast: bool = False,
    output_dir: str | Path | None = None,
    qa: bool = True,
    save: bool = True,
    load_raw: bool = True,
    interactive: bool = False,
    log: Callable[[str], None] | None = print,
) -> ReductionState:
    """Run the onfire reduction pipeline over the canonical step sequence.

    Args:
        frames: Path to the fixed-width Frame Table; defaults to the ``frames``
            value of the Parameter Config.
        config: Optional path to the YAML Parameter Config.
        until: Stop after completing this step (name or ordinal).
        from_step: Resume from the checkpoint of the preceding step and begin
            executing at this step (name or ordinal).
        fail_fast: When ``True``, halt on the first step error or masked order,
            saving the state and raising :class:`FailFastError` with the exact
            code needed to reproduce the failure.
        output_dir: Directory for checkpoints; defaults to the config value.
        qa: When ``True``, save Plotly QA HTML artifacts after each step.
        save: When ``True``, write a FITS checkpoint after each step.
        load_raw: When ``True``, read each step's raw exposures before running it.
        interactive: When ``True``, drop into an IPython session bound to the
            final state after the requested steps complete.
        log: Optional callback receiving progress messages; ``None`` is silent.

    Returns:
        The fully reduced :class:`~onfire.pipeline.state.ReductionState`.

    Raises:
        FailFastError: If ``fail_fast`` is enabled and a step fails.
        FileNotFoundError: If ``from_step`` needs a checkpoint that is missing.
    """
    start = _resolve_ordinal(from_step)
    stop = _resolve_ordinal(until)

    if frames is None:
        frames = load_config(config).get("frames", "frames.ascii")
    config_state = ReductionState.from_files(frames, config)
    resolved_output = Path(output_dir or config_state.config.get("output_dir", "checkpoints"))
    qa_dir = Path(config_state.config.get("qa_dir", "qa"))

    if start is not None and start > 1:
        previous = _checkpoint_path(resolved_output, start - 1)
        if not previous.exists():
            raise FileNotFoundError(
                f"cannot resume from {step_name(start)!r}: missing checkpoint {previous}"
            )
        state = ReductionState.load(previous)
        _log(log, f"resumed from {previous}")
    else:
        start = 1
        state = config_state

    if stop is None:
        stop = len(STEP_SEQUENCE)
    if stop < start:
        raise ValueError(
            f"--until ({step_name(stop)}) precedes --from ({step_name(start)})"
        )

    for ordinal in range(start, stop + 1):
        name = step_name(ordinal)
        if load_raw:
            state.load_raw_frames(STEP_ROLES.get(name, ()))
            if name == "extinction_correct":
                _attach_target_coordinates(state)
        resolved = state.resolve_call_params(name)
        masked_before = set(state.order_mask.masked_orders)
        snapshot = state.clone() if fail_fast else None
        try:
            state = state.run_step(name, qa=qa)
        except Exception as exc:  # noqa: BLE001 - re-raised for both modes
            if fail_fast:
                checkpoint = _persist(snapshot, resolved_output, qa_dir, qa, save)
                code = format_reproduction_code(
                    name, resolved, checkpoint=checkpoint, qa=qa
                )
                raise FailFastError(
                    f"step {name!r} failed: {exc}",
                    step=name,
                    state=snapshot,
                    params=resolved,
                    order=None,
                    checkpoint=checkpoint,
                    reproduction_code=code,
                ) from exc
            raise
        checkpoint = _persist(state, resolved_output, qa_dir, qa, save)
        _log(log, f"{ordinal:02d} {name} -> {checkpoint}")

        newly_masked = state.order_mask.masked_orders - masked_before
        if fail_fast and newly_masked:
            order = min(newly_masked)
            previous = _persist(snapshot, resolved_output, qa_dir, qa, save)
            code = format_reproduction_code(
                name, resolved, checkpoint=previous, order=order, qa=qa
            )
            raise FailFastError(
                f"step {name!r} masked order(s) {sorted(newly_masked)}: "
                f"{state.order_mask.reasons.get(order, '')}",
                step=name,
                state=snapshot,
                params=resolved,
                order=order,
                checkpoint=previous,
                reproduction_code=code,
            )

    if interactive:
        _launch_interactive(state)
    return state


def _persist(
    state: ReductionState,
    output_dir: Path,
    qa_dir: Path,
    qa: bool,
    write: bool,
) -> Optional[Path]:
    """Persist the state checkpoint and QA artifacts, returning the checkpoint."""
    checkpoint: Optional[Path] = None
    if write:
        checkpoint = state.save(output_dir)
    if qa and state.plots:
        state.save_plots(qa_dir)
    return checkpoint


def _attach_target_coordinates(state: ReductionState) -> None:
    """Populate ``meta['ra']``/``meta['dec']`` from the science FITS header."""
    coordinates = read_target_coordinates(state)
    if coordinates is None:
        return
    ra, dec = coordinates
    state.meta.setdefault("ra", ra)
    state.meta.setdefault("dec", dec)


def _launch_interactive(state: ReductionState) -> None:
    """Drop into an interactive session with the reduced ``state`` in scope."""
    banner = (
        "onfire interactive session: the reduced ReductionState is bound to "
        "'state'; run state.run_step(...) to continue interactively."
    )
    namespace = {"state": state}
    try:
        from IPython import embed
    except ImportError:  # pragma: no cover - IPython is normally present
        import code

        code.interact(banner=banner, local=namespace)
    else:  # pragma: no cover - interactive by nature
        embed(banner1=banner, user_ns=namespace)


run = reduce
"""Alias for :func:`reduce`, the top-level orchestration runner."""

__all__ = [
    "FailFastError",
    "STEP_ROLES",
    "format_reproduction_code",
    "reduce",
    "run",
]
