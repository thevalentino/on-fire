"""Purpose-built comparison figures for reduction steps.

The default :meth:`~onfire.pipeline.state.ReductionState.compare` pairs two
states' QA Artifacts. A step whose corrections deserve a bespoke layout registers
one of these builders with
:func:`~onfire.pipeline.steps.register_step_comparison`. Builders live in the
pipeline tier because they read a :class:`~onfire.pipeline.state.ReductionState`;
the Step Engine tier stays pure (ADR 0003).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from plotly import graph_objects as go

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onfire.pipeline.state import ReductionState

# The atmospheric corrections a state can hold as candidates, in display order.
TELLURIC_METHODS: tuple[str, ...] = ("empirical", "grid")


def telluric_candidate_methods(state: "ReductionState") -> list[str]:
    """Return the telluric correction methods ``state`` holds candidate products for."""
    return [
        method
        for method in TELLURIC_METHODS
        if state.products.get(f"telluric_{method}_transmission") is not None
    ]


def _add_telluric_trace(
    figure: go.Figure,
    wave: Any,
    values: Any,
    name: str,
    *,
    row: int,
    column: int,
) -> None:
    """Add one transmission or corrected-spectrum trace for the first object."""
    values = np.asarray(values, dtype=float)
    x = (
        np.asarray(wave, dtype=float)[:, 0]
        if wave is not None
        else np.arange(values.shape[0])
    )
    figure.add_trace(
        go.Scatter(x=x, y=values[:, 0], mode="lines", name=name),
        row=row,
        col=column,
    )


def _add_telluric_state(
    figure: go.Figure,
    state: "ReductionState",
    methods: list[str],
    *,
    column: int,
    label: str,
) -> None:
    """Plot every available correction for one state into a single column."""
    wave = state.products.get("telluric_wave")
    for method in methods:
        _add_telluric_trace(
            figure,
            wave,
            state.products[f"telluric_{method}_transmission"],
            f"{label}{method} transmission",
            row=1,
            column=column,
        )
        _add_telluric_trace(
            figure,
            wave,
            state.products[f"telluric_{method}_flux"],
            f"{label}{method} corrected",
            row=2,
            column=column,
        )


def telluric_comparison(
    state: "ReductionState", other: "ReductionState"
) -> go.Figure:
    """Build the purpose-built Telluric correction comparison figure.

    Registered as the telluric step's comparison so ``compare`` keeps a layout
    made for the two corrections (empirical and grid) instead of pairing QA
    Artifacts blindly.

    With two distinct states it renders a 2x2 figure -- one column per state, a
    transmission row over a corrected-spectrum row, each carrying every available
    correction. With a single state (``other is state``, used by the
    ``compare_telluric`` entry point) it renders one column per correction.

    Raises:
        ValueError: If a state holds no telluric candidate products.
    """
    from plotly.subplots import make_subplots

    if other is state:
        methods = telluric_candidate_methods(state)
        if not methods:
            raise ValueError(
                "compare_telluric requires telluric candidate products; "
                "run telluric_correct first"
            )
        figure = make_subplots(
            rows=2,
            cols=len(methods),
            subplot_titles=[f"{method} transmission" for method in methods]
            + [f"{method} corrected" for method in methods],
            specs=[[{}] * len(methods), [{}] * len(methods)],
        )
        wave = state.products.get("telluric_wave")
        for column, method in enumerate(methods, start=1):
            _add_telluric_trace(
                figure,
                wave,
                state.products[f"telluric_{method}_transmission"],
                f"{method} transmission",
                row=1,
                column=column,
            )
            _add_telluric_trace(
                figure,
                wave,
                state.products[f"telluric_{method}_flux"],
                f"{method} corrected",
                row=2,
                column=column,
            )
    else:
        columns = (("this state", state), ("the other state", other))
        methods = []
        for label, candidate_state in columns:
            available = telluric_candidate_methods(candidate_state)
            if not available:
                raise ValueError(
                    f"{label} has no telluric candidate products; "
                    "run telluric_correct first"
                )
            methods.append(available)
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[f"{label} transmission" for label, _ in columns]
            + [f"{label} corrected" for label, _ in columns],
            specs=[[{}] * 2, [{}] * 2],
        )
        for column, ((label, candidate_state), available) in enumerate(
            zip(columns, methods), start=1
        ):
            _add_telluric_state(
                figure, candidate_state, available, column=column, label=f"{label} "
            )

    figure.update_layout(
        title="Telluric correction comparison",
        height=700,
        showlegend=False,
    )
    figure.update_xaxes(title_text="wavelength (Angstrom)")
    return figure
