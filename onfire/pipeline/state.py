"""The ReductionState container: the central data model and interactive boundary.

:class:`ReductionState` orchestrates Step Engine execution, persists intermediate
data as Multi-Extension FITS checkpoints, provides in-memory State Cloning for
experimental parameter exploration, and collects interactive Plotly QA figures.
The computational algorithms themselves stay in :mod:`onfire.core`; this module
only manages state (ADR 0003).
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
from astropy.table import Table

from onfire.core.fire import FIRE_ORDER_NUMBERS
from onfire.setup.table import FrameTable

from onfire.pipeline.config import load_config
from onfire.pipeline.steps import (
    CROSS_ORDER_STEPS,
    StepResult,
    get_step_comparison,
    get_step_engine,
    step_name,
    step_ordinal,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plotly.graph_objects import Figure


@dataclass
class OrderMask:
    """Boolean status flags marking echelle orders as unusable, with reasons."""

    reasons: dict[int, str] = field(default_factory=dict)

    def mask(self, order: int, reason: str = "") -> None:
        """Flag ``order`` as unusable, recording an optional reason."""
        self.reasons[int(order)] = reason

    def unmask(self, order: int) -> None:
        """Clear any mask on ``order``."""
        self.reasons.pop(int(order), None)

    def is_masked(self, order: int) -> bool:
        return int(order) in self.reasons

    @property
    def masked_orders(self) -> set[int]:
        return set(self.reasons)

    def copy(self) -> "OrderMask":
        return OrderMask(reasons=dict(self.reasons))

    def to_dict(self) -> dict[str, str]:
        return {str(order): reason for order, reason in self.reasons.items()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "OrderMask":
        return cls(reasons={int(order): reason for order, reason in data.items()})


@dataclass
class StaleMarks:
    """Reduction steps whose products are out of date, scoped to affected orders.

    A step that is absent is up to date. A present step is Stale either for every
    order or for a specific set of orders. The "every order" encoding is private
    to this type, so callers only ask whether a given order is Stale.
    """

    _marks: dict[int, Optional[set[int]]] = field(default_factory=dict)

    def __contains__(self, step: int) -> bool:
        return int(step) in self._marks

    def steps(self) -> set[int]:
        """Return the ordinals of every step currently marked Stale."""
        return set(self._marks)

    def is_stale_order(self, step: int, order: int) -> bool:
        """Whether ``order``'s products of ``step`` are Stale."""
        marked = self._marks.get(int(step))
        if marked is None:
            return int(step) in self._marks
        return int(order) in marked

    def mark_every_order(self, step: int) -> None:
        """Mark every product of ``step`` Stale."""
        self._marks[int(step)] = None

    def mark_order(self, step: int, order: int) -> None:
        """Mark ``order``'s products of ``step`` Stale; a whole-step mark stays."""
        step, order = int(step), int(order)
        if step not in self._marks:
            self._marks[step] = {order}
        elif self._marks[step] is not None:
            self._marks[step].add(order)

    def clear(self, step: int) -> None:
        """Clear every Stale mark on ``step``."""
        self._marks.pop(int(step), None)

    def clear_order(self, step: int, order: int) -> None:
        """Clear ``order``'s Stale mark on ``step``, leaving a whole-step mark."""
        step, order = int(step), int(order)
        marked = self._marks.get(step)
        if marked is not None:
            marked.discard(order)
            if not marked:
                del self._marks[step]

    def to_json(self) -> dict[str, Any]:
        """Serialise to the checkpoint JSON shape (``null`` means every order)."""
        return {
            str(step): None if orders is None else sorted(orders)
            for step, orders in self._marks.items()
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StaleMarks":
        """Rebuild from the checkpoint JSON shape written by :meth:`to_json`."""
        return cls(
            _marks={
                int(step): None if orders is None else {int(o) for o in orders}
                for step, orders in data.items()
            }
        )


@dataclass(frozen=True)
class ProductProvenance:
    """Which step produced a product and, where it has one, its order axis.

    ``orders`` lists the echelle orders a product carries, in the same order as
    they appear along its order axis (or its ``order`` column). ``None`` marks a
    product with no order axis -- whole-state data discarded whole when Stale.
    """

    step: int
    orders: Optional[tuple[int, ...]] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "orders": None if self.orders is None else list(self.orders),
        }

    @classmethod
    def from_json(cls, data: Any) -> "ProductProvenance":
        """Rebuild from checkpoint JSON.

        A bare integer is the legacy record shape and carries no order axis, so
        it loads as a whole-state product.
        """
        if isinstance(data, int):
            return cls(step=int(data))
        orders = data.get("orders")
        return cls(
            step=int(data["step"]),
            orders=None if orders is None else tuple(int(o) for o in orders),
        )


@dataclass
class Provenance:
    """Which reduction step produced each product and each QA Artifact.

    Each product records the orders it covers, where it has an order axis;
    products with no order axis are whole-state. Each step also records the
    orders its QA Artifacts cover, so a missing artifact can name a lone
    single-order run.
    """

    products: dict[str, ProductProvenance] = field(default_factory=dict)
    plots: dict[str, int] = field(default_factory=dict)
    plot_orders: dict[int, Optional[tuple[int, ...]]] = field(default_factory=dict)

    def steps(self) -> set[int]:
        """Return every step ordinal that has recorded a product or QA Artifact."""
        return {
            *(record.step for record in self.products.values()),
            *self.plots.values(),
        }

    def plot_coverage(self, ordinal: int) -> Optional[tuple[int, ...]]:
        """Return the orders ``ordinal``'s QA Artifacts cover, or ``None``.

        ``None`` means the step covers every order, or that no coverage was
        recorded; either way the caller should not name a single order.
        """
        return self.plot_orders.get(int(ordinal))

    def record_plot_coverage(self, ordinal: int, order: Optional[int]) -> None:
        """Record the orders ``ordinal``'s QA Artifacts cover.

        A full run (``order`` is ``None``) covers every order. A single-order run
        adds that order, unless the step is already recorded as whole-state.
        Coverage is recorded for every run, so a step run with QA off still
        explains which order it targeted.
        """
        ordinal = int(ordinal)
        if order is None:
            self.plot_orders[ordinal] = None
            return
        if ordinal in self.plot_orders and self.plot_orders[ordinal] is None:
            return
        orders = set(self.plot_orders.get(ordinal) or ())
        orders.add(int(order))
        self.plot_orders[ordinal] = tuple(sorted(orders))

    def drop_plot_coverage(self, steps: Iterable[int]) -> None:
        """Forget QA Artifact order coverage for each step in ``steps``."""
        for step in steps:
            self.plot_orders.pop(int(step), None)

    def record_products(
        self, ordinal: int, coverage: dict[str, Optional[tuple[int, ...]]]
    ) -> None:
        """Record ``ordinal`` as producer of each product and the orders it covers."""
        for key, orders in coverage.items():
            self.products[key] = ProductProvenance(step=ordinal, orders=orders)

    def record_plots(self, ordinal: int, names: Iterable[str]) -> None:
        """Record ``ordinal`` as the producer of each named QA Artifact."""
        for name in names:
            self.plots[name] = ordinal

    def to_json(self) -> dict[str, Any]:
        """Serialise product provenance to the checkpoint JSON shape."""
        return {key: record.to_json() for key, record in self.products.items()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Provenance":
        """Rebuild product provenance from the checkpoint JSON shape."""
        return cls(
            products={
                key: ProductProvenance.from_json(value) for key, value in data.items()
            }
        )


@dataclass
class Lineage:
    """Session-local identity and revision counters for a state's clone graph.

    A State Clone remembers its parent's identity and the parent revision it was
    cloned at, so a Commit can refuse a clone whose Baseline has moved on. The
    counters are never persisted, so each loaded state is its own lineage.
    """

    uid: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_uid: Optional[str] = None
    base_revision: int = 0
    revision: int = 0


def _order_axis_index(value: np.ndarray, size: int) -> Optional[int]:
    """Return the array's sole axis of length ``size``, or ``None`` if ambiguous."""
    axes = [axis for axis, length in enumerate(value.shape) if length == size]
    return axes[0] if len(axes) == 1 else None


def _order_coverage(value: Any) -> Optional[tuple[int, ...]]:
    """Return orders a freshly produced ``value`` covers, or ``None`` if whole-state.

    A product has an order axis when it is an Astropy table with an ``order``
    column or a NumPy array with a single dimension matching the canonical FIRE
    orders. Every other product is whole-state.
    """
    if isinstance(value, Table):
        if "order" in value.colnames:
            return tuple(int(order) for order in value["order"])
        return None
    if isinstance(value, np.ndarray):
        if _order_axis_index(value, FIRE_ORDER_NUMBERS.size) is None:
            return None
        return tuple(int(order) for order in FIRE_ORDER_NUMBERS)
    return None


def _trim_stale_orders(
    value: Any,
    coverage: Optional[tuple[int, ...]],
    stale: "StaleMarks",
    step: int,
) -> tuple[Any, Optional[tuple[int, ...]]]:
    """Drop the Stale orders from an order-scoped product.

    Returns ``(kept_value, kept_orders)``. Returns ``(None, None)`` when the whole
    product must be discarded: it is whole-state, its order axis no longer
    matches its recorded coverage, or every covered order is Stale.
    """
    if coverage is None or value is None:
        return None, None
    if isinstance(value, Table) and "order" in value.colnames:
        orders = np.asarray(value["order"], dtype=int)
        keep = np.array(
            [not stale.is_stale_order(step, int(order)) for order in orders],
            dtype=bool,
        )
        if not keep.any():
            return None, None
        return value[keep], tuple(int(order) for order in orders[keep])
    if isinstance(value, np.ndarray):
        axis = _order_axis_index(value, len(coverage))
        if axis is None:
            return None, None
        keep_indices = [
            index
            for index, order in enumerate(coverage)
            if not stale.is_stale_order(step, order)
        ]
        if not keep_indices:
            return None, None
        kept = np.take(value, keep_indices, axis=axis)
        return kept, tuple(coverage[index] for index in keep_indices)
    return None, None


def _missing_artifact_message(
    label: str, name: str, coverage: Optional[tuple[int, ...]]
) -> str:
    """Explain a missing QA Artifact, naming a lone single-order run's order.

    A single-order run only produces that order's artifact, so naming the order
    explains a figure that looks missing. A full run (or one with no recorded
    coverage) gets the plain instruction to re-run with QA on.
    """
    message = f"{label} has no QA Artifact for step {name!r}; run the step with qa=True"
    if coverage is not None and len(coverage) == 1:
        return (
            f"{message} (it was run for order {coverage[0]} alone, whose artifact "
            "a single-order run produces)"
        )
    return message


@dataclass
class ReductionState:
    """Central container for intermediate reduction data and step execution.

    Attributes:
        frames: The input Frame Table describing raw exposures and their roles.
        config: The resolved Parameter Config.
        products: Named data products (NumPy arrays or Astropy tables).
        meta: Named scalar metadata such as fit statistics.
        step: The ordinal of the last completed reduction step (0 = none).
        order_mask: Flags marking unusable orders.
        plots: Live Plotly figures attached as QA artifacts, keyed by name.
        source: Provenance string, typically the Frame Table path.
        provenance: Which step produced each product and the orders it covers,
            and which step and orders produced each attached QA artifact.
        stale: Steps whose products are out of date because an earlier step was
            re-run, scoped to the affected orders. Stale products are kept until
            a Commit discards them.
        lineage: Session-local identity and revision counters for State Clones.
    """

    frames: FrameTable
    config: dict[str, Any] = field(default_factory=dict)
    products: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    order_mask: OrderMask = field(default_factory=OrderMask)
    plots: dict[str, "Figure"] = field(default_factory=dict)
    source: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    stale: StaleMarks = field(default_factory=StaleMarks)
    lineage: Lineage = field(default_factory=Lineage, repr=False, compare=False)

    @classmethod
    def from_files(
        cls,
        frames_path: str | Path,
        config_path: str | Path | None = None,
    ) -> "ReductionState":
        """Build a state from a Frame Table file and an optional YAML config."""
        frames_path = Path(frames_path)
        return cls(
            frames=FrameTable.read(frames_path),
            config=load_config(config_path),
            source=str(frames_path),
        )

    def load_raw_frames(
        self, roles: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Read raw exposures listed in the Frame Table into :attr:`products`.

        Raw FITS reading lives in the persistence tier; the resulting stacks are
        stored under keys like ``raw_flats`` for the Step Engines to consume.

        Args:
            roles: Frame Table roles to load (e.g. ``["pixelflat"]``). When
                ``None``, every known role is attempted.

        Returns:
            The mapping of product key to stacked raw array that was loaded.
        """
        from onfire.pipeline.raw import read_raw_frames

        return read_raw_frames(self, roles)

    @property
    def step_name(self) -> str:
        """Name of the last completed reduction step, or '' if none."""
        return step_name(self.step)

    def checkpoint_filename(self) -> str:
        """Chronologically prefixed Multi-Extension FITS checkpoint filename."""
        if self.step <= 0:
            return "00_initial.fits"
        return f"{self.step:02d}_{self.step_name}.fits"

    def clone(self) -> "ReductionState":
        """Return an independent State Clone for experimental parameter changes.

        The clone holds its own data products, metadata, Order Mask, and Plotly
        figures, so mutating them never touches this Baseline state.
        """
        clone = copy.deepcopy(self)
        clone.lineage = Lineage(
            parent_uid=self.lineage.uid,
            base_revision=self.lineage.revision,
        )
        return clone

    def commit(self, clone: "ReductionState", *, force: bool = False) -> None:
        """Replace this Baseline wholesale with a State Clone.

        The Baseline object is mutated in place and the ``clone`` is left
        unchanged. Products still marked Stale on the clone are discarded and the
        recorded progress is rewound to the last step with no Stale products.
        Nothing is written to disk.

        Args:
            clone: A State Clone made from this state with :meth:`clone` or
                ``run_step(..., clone=True)``.
            force: Commit even if the clone is from an unrelated state or this
                Baseline has changed since the clone was made.

        Raises:
            ValueError: If the clone was not made from this state, or this
                Baseline has diverged since, and ``force`` is not set.
        """
        if not force:
            if clone.lineage.parent_uid != self.lineage.uid:
                raise ValueError(
                    "cannot commit: the clone was not made from this state "
                    "(unrelated lineage); pass force=True to commit anyway"
                )
            if clone.lineage.base_revision != self.lineage.revision:
                raise ValueError(
                    "cannot commit: this Baseline has diverged since the clone "
                    "was made; pass force=True to commit anyway"
                )
        snapshot = copy.deepcopy(clone)
        snapshot._discard_stale()
        self._adopt(snapshot)
        self.lineage.revision += 1

    def _adopt(self, other: "ReductionState") -> None:
        """Replace this state's data with ``other``'s, keeping its own lineage."""
        for data_field in fields(self):
            if data_field.name != "lineage":
                setattr(self, data_field.name, getattr(other, data_field.name))

    def _drop_qa_artifacts(self, steps: set[int]) -> None:
        """Drop every attached QA Artifact produced by a step in ``steps``."""
        for name, step in list(self.provenance.plots.items()):
            if step in steps:
                self.plots.pop(name, None)
                del self.provenance.plots[name]
        self.provenance.drop_plot_coverage(steps)

    def _discard_stale(self) -> None:
        """Drop every Stale product and QA artifact, rewinding progress."""
        for key, record in list(self.provenance.products.items()):
            if record.step not in self.stale:
                continue
            value, kept_orders = _trim_stale_orders(
                self.products.get(key), record.orders, self.stale, record.step
            )
            if value is None:
                self.products.pop(key, None)
                del self.provenance.products[key]
            else:
                self.products[key] = value
                self.provenance.products[key] = ProductProvenance(
                    step=record.step, orders=kept_orders
                )
        stale_steps = self.stale.steps()
        self._drop_qa_artifacts(stale_steps)
        if stale_steps:
            first_stale = min(stale_steps)
            remaining = self.provenance.steps()
            self.step = min(
                self.step, max((s for s in remaining if s < first_stale), default=0)
            )
        self.stale = StaleMarks()

    def _record_run(
        self, ordinal: int, result: StepResult, order: Optional[int], qa: bool
    ) -> None:
        """Record provenance for a step run and mark later steps Stale."""
        later = {step for step in self.provenance.steps() if step > ordinal}
        for step in later:
            if order is None or step_name(step) in CROSS_ORDER_STEPS:
                self.stale.mark_every_order(step)
            else:
                self.stale.mark_order(step, int(order))
        if ordinal in self.stale:
            if order is None:
                self.stale.clear(ordinal)
            else:
                self.stale.clear_order(ordinal, int(order))
        coverage = {
            key: _order_coverage(value) for key, value in result.products.items()
        }
        self.provenance.record_products(ordinal, coverage)
        self.provenance.record_plot_coverage(ordinal, order)
        if qa:
            self.provenance.record_plots(ordinal, result.plots)

    def compare(self, other: "ReductionState", step: str | int) -> "Figure":
        """Compare this state with ``other`` for one reduction step.

        By default this pairs each state's QA Artifacts for ``step`` into a dual
        Plotly figure, this state on the left and ``other`` on the right. A step
        may register its own builder with
        :func:`~onfire.pipeline.steps.register_step_comparison`.

        Raises:
            ValueError: If either state has no QA Artifact for ``step``.
        """
        from plotly.subplots import make_subplots

        name = step if isinstance(step, str) else step_name(step)
        builder = get_step_comparison(name)
        if builder is not None:
            return builder(self, other)

        ordinal = step_ordinal(name)
        sides = (("this state", self), ("the other state", other))
        columns = []
        for label, state in sides:
            figures = [
                state.plots[plot]
                for plot, plot_step in state.provenance.plots.items()
                if plot_step == ordinal and plot in state.plots
            ]
            if not figures:
                coverage = state.provenance.plot_coverage(ordinal)
                raise ValueError(_missing_artifact_message(label, name, coverage))
            columns.append(figures)

        figure = make_subplots(
            rows=1, cols=2, subplot_titles=[f"{label}: {name}" for label, _ in sides]
        )
        for column, figures in enumerate(columns, start=1):
            for qa_figure in figures:
                for trace in qa_figure.data:
                    figure.add_trace(trace, row=1, col=column)
        figure.update_layout(title=f"Comparison: {name}")
        return figure

    def add_plot(self, name: str, figure: "Figure") -> None:
        """Attach an interactive Plotly ``figure`` to :attr:`plots`."""
        self.plots[name] = figure

    def compare_telluric(
        self, select: Optional[str] = None, *, commit: Optional[str] = None
    ) -> "Figure":
        """Compare the empirical and grid telluric corrections interactively.

        Renders a dual Plotly figure showing, for the first reduced object, the
        transmission and the corrected spectrum of each available correction.
        When ``select`` names a method (``"empirical"`` or ``"grid"``), that
        candidate's transmission, flux, variance, inverse variance, and mask are
        copied to the canonical ``telluric_*`` products and the ``telluric``
        table's method column is updated.

        Args:
            select: Method to select into the state, or ``None`` to only render.
            commit: Deprecated alias of ``select``; "commit" now means replacing the
                Baseline with a State Clone (see :meth:`commit`).

        Returns:
            The comparison :class:`~plotly.graph_objects.Figure`, also attached
            to :attr:`plots` under ``"telluric_comparison"``.

        Raises:
            ValueError: If no telluric candidates exist, or ``select`` names an
                unavailable method.
        """
        import warnings

        from plotly import graph_objects as go
        from plotly.subplots import make_subplots

        if commit is not None:
            warnings.warn(
                "compare_telluric(commit=...) is deprecated; use select=...",
                DeprecationWarning,
                stacklevel=2,
            )
            select = commit if select is None else select

        candidates = [
            method
            for method in ("empirical", "grid")
            if self.products.get(f"telluric_{method}_transmission") is not None
        ]
        if not candidates:
            raise ValueError(
                "compare_telluric requires telluric candidate products; "
                "run telluric_correct first"
            )
        if select is not None:
            select = str(select).lower()
            if select not in candidates:
                raise ValueError(
                    f"cannot select {select!r}; available methods: {candidates}"
                )
            for suffix in ("transmission", "flux", "var", "ivar", "mask"):
                self.products[f"telluric_{suffix}"] = self.products[
                    f"telluric_{select}_{suffix}"
                ]
            table = self.products.get("telluric")
            if table is not None:
                table["method"] = select
            self.meta["telluric_method"] = select

        wave = self.products.get("telluric_wave")
        wave = None if wave is None else np.asarray(wave, dtype=float)
        figure = make_subplots(
            rows=2,
            cols=len(candidates),
            subplot_titles=[f"{method} transmission" for method in candidates]
            + [f"{method} corrected" for method in candidates],
            specs=[[{}] * len(candidates), [{}] * len(candidates)],
        )
        for column, method in enumerate(candidates, start=1):
            transmission = np.asarray(
                self.products[f"telluric_{method}_transmission"], dtype=float
            )
            corrected = np.asarray(
                self.products[f"telluric_{method}_flux"], dtype=float
            )
            x = wave[:, 0] if wave is not None else np.arange(transmission.shape[0])
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=transmission[:, 0],
                    mode="lines",
                    name=f"{method} transmission",
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=corrected[:, 0],
                    mode="lines",
                    name=f"{method} corrected",
                ),
                row=2,
                col=column,
            )
        figure.update_layout(
            title="Telluric correction comparison",
            height=700,
            showlegend=False,
        )
        figure.update_xaxes(title_text="wavelength (Angstrom)")
        self.add_plot("telluric_comparison", figure)
        return figure

    def resolve_call_params(
        self,
        step: str | int,
        *,
        params: Optional[dict[str, Any]] = None,
        order: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return the effective parameters a step would be called with.

        The Parameter Config seeds the call (top-level control keys, then the
        step-specific ``steps.<name>`` block), explicit ``params`` override it,
        and a targeted ``order`` is added last. Pipeline-control keys
        (``steps``, ``qa_dir``, ``frames``, ``output_dir``) are not algorithm
        inputs and are never forwarded.
        """
        name = step if isinstance(step, str) else step_name(step)
        call_params: dict[str, Any] = {
            key: value
            for key, value in self.config.items()
            if key not in ("steps", "qa_dir", "frames", "output_dir")
        }
        call_params.update(self.config.get("steps", {}).get(name, {}) or {})
        call_params.update(params or {})
        if order is not None:
            call_params.setdefault("order", order)
        return call_params

    def run_step(
        self,
        step: str | int,
        *,
        params: Optional[dict[str, Any]] = None,
        clone: bool = False,
        qa: bool = True,
        order: Optional[int] = None,
    ) -> "ReductionState":
        """Execute a reduction step, in place or on an isolated State Clone.

        Args:
            step: Step name or 1-based ordinal in the canonical sequence.
            params: Algorithmic parameter overrides merged over the config.
            clone: When ``True``, run on and return an independent State Clone,
                leaving this baseline state untouched.
            qa: When ``True``, attach any returned Plotly figures to :attr:`plots`.
            order: Target a single echelle order (interactive single-order runs).

        Returns:
            The mutated baseline state, or the new State Clone when ``clone`` is
            ``True``.
        """
        name = step if isinstance(step, str) else step_name(step)
        ordinal = step_ordinal(name)
        engine = get_step_engine(name)

        call_params = self.resolve_call_params(step, params=params, order=order)

        target = self.clone() if clone else self
        result = engine(target.products, target.meta, call_params)
        if not isinstance(result, StepResult):
            raise TypeError(
                f"engine for {name!r} returned {type(result).__name__}, "
                "expected StepResult"
            )

        target.products.update(result.products)
        target.meta.update(result.meta)
        if order is None:
            # A full run supersedes the step's earlier QA Artifacts; a
            # single-order run only replaces the artifacts it re-creates.
            target._drop_qa_artifacts({ordinal})
        if qa:
            target.plots.update(result.plots)
        if order is not None and int(order) not in result.masked_orders:
            # A successful targeted re-run is evidence the old mask no longer applies.
            target.order_mask.unmask(order)
        for masked_order, reason in result.masked_orders.items():
            target.order_mask.mask(masked_order, reason)
        # Keep the high-water mark: re-running an earlier step (e.g. a single
        # order) must not rewind the recorded progress.
        target.step = max(target.step, ordinal)
        target._record_run(ordinal, result, order, qa)
        target.lineage.revision += 1
        return target

    def save(self, directory: str | Path, filename: Optional[str] = None) -> Path:
        """Write a Multi-Extension FITS checkpoint into ``directory``.

        Returns:
            The path of the written checkpoint.
        """
        from onfire.pipeline.fits import write_checkpoint

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (filename or self.checkpoint_filename())
        write_checkpoint(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReductionState":
        """Reload a state previously written by :meth:`save` without loss."""
        from onfire.pipeline.fits import read_checkpoint

        return read_checkpoint(path)

    def save_plots(
        self,
        directory: str | Path,
        prefix: Optional[str] = None,
    ) -> list[Path]:
        """Write every attached Plotly figure as a standalone HTML QA file.

        Args:
            directory: Destination directory for the HTML artifacts.
            prefix: Filename prefix; defaults to the checkpoint stem.

        Returns:
            The paths of the written HTML files, in insertion order.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = prefix or Path(self.checkpoint_filename()).stem
        written: list[Path] = []
        for name, figure in self.plots.items():
            path = directory / f"{prefix}_{name}.html"
            figure.write_html(str(path))
            written.append(path)
        return written
