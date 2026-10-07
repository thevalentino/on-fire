"""Reduction Step registry and the pure Step Engine contract.

Step Engines live in :mod:`onfire.core` and operate purely on plain NumPy
arrays, Astropy tables, and numeric dictionaries. They never see a
:class:`~onfire.pipeline.state.ReductionState`; instead they are registered here
by name and invoked by the state container, keeping computation decoupled from
persistence (ADR 0003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# The canonical 12-step reduction sequence, in execution order.
STEP_SEQUENCE: tuple[str, ...] = (
    "prep_calibrations",
    "trace_orders",
    "calibrate_wavelengths",
    "fit_tilts",
    "process_science",
    "find_objects",
    "subtract_sky",
    "extract_1d",
    "flux_calibrate",
    "telluric_correct",
    "coadd_orders",
    "extinction_correct",
)


def step_ordinal(name: str) -> int:
    """Return the 1-based position of ``name`` in the reduction sequence."""
    try:
        return STEP_SEQUENCE.index(name) + 1
    except ValueError:
        raise ValueError(
            f"unknown reduction step {name!r}; expected one of {STEP_SEQUENCE}"
        ) from None


def step_name(ordinal: int) -> str:
    """Return the step name for a 1-based ``ordinal``, or '' for 0."""
    if ordinal == 0:
        return ""
    if 1 <= ordinal <= len(STEP_SEQUENCE):
        return STEP_SEQUENCE[ordinal - 1]
    raise ValueError(f"step ordinal {ordinal} out of range 0..{len(STEP_SEQUENCE)}")


@dataclass
class StepResult:
    """The output of executing a Step Engine.

    Attributes:
        products: Named data products to merge into the state (arrays or tables).
        meta: Named scalar metadata to merge into the state.
        plots: Named Plotly figures produced as QA artifacts.
        masked_orders: Orders to flag unusable, mapped to a human-readable reason.
    """

    products: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    masked_orders: dict[int, str] = field(default_factory=dict)


class StepEngine(Protocol):
    """A pure reduction algorithm called by :meth:`ReductionState.run_step`."""

    def __call__(
        self,
        products: dict[str, Any],
        meta: dict[str, Any],
        params: dict[str, Any],
    ) -> StepResult: ...


_REGISTRY: dict[str, StepEngine] = {}
_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {}


def register_step_engine(name: str, engine: StepEngine) -> None:
    """Register the engine that implements reduction step ``name``."""
    step_ordinal(name)  # validates the name against the canonical sequence
    _REGISTRY[name] = engine


def get_step_engine(name: str) -> StepEngine:
    """Return the engine registered for ``name``.

    Raises:
        KeyError: If no engine has been registered for the step.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no engine registered for reduction step {name!r}") from None


def registered_steps() -> tuple[str, ...]:
    """Return the names of all currently registered Step Engines."""
    return tuple(name for name in STEP_SEQUENCE if name in _REGISTRY)


def register_step_comparison(name: str, builder: Callable[[Any, Any], Any]) -> None:
    """Register a bespoke comparison figure builder for reduction step ``name``.

    ``builder(state, other)`` receives the two states being compared and returns
    a Plotly figure. It replaces the default QA Artifact pairing for that step.
    """
    step_ordinal(name)  # validates the name against the canonical sequence
    _COMPARISONS[name] = builder


def get_step_comparison(name: str) -> Callable[[Any, Any], Any] | None:
    """Return the registered comparison builder for ``name``, if any."""
    return _COMPARISONS.get(name)


def clear_step_engines() -> None:
    """Remove all registered Step Engines and comparisons, for test isolation."""
    _REGISTRY.clear()
    _COMPARISONS.clear()
