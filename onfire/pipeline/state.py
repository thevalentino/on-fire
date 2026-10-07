"""The ReductionState container: the central data model and interactive boundary.

:class:`ReductionState` orchestrates Step Engine execution, persists intermediate
data as Multi-Extension FITS checkpoints, provides in-memory State Cloning for
experimental parameter exploration, and collects interactive Plotly QA figures.
The computational algorithms themselves stay in :mod:`onfire.core`; this module
only manages state (ADR 0003).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from onfire.setup.table import FrameTable

from onfire.pipeline.config import load_config
from onfire.pipeline.steps import (
    StepResult,
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
    """

    frames: FrameTable
    config: dict[str, Any] = field(default_factory=dict)
    products: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    order_mask: OrderMask = field(default_factory=OrderMask)
    plots: dict[str, "Figure"] = field(default_factory=dict)
    source: str = ""

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

        The clone is a deep copy, so mutating its data products, metadata, order
        mask, or Plotly figures never touches this baseline state.
        """
        return copy.deepcopy(self)

    def add_plot(self, name: str, figure: "Figure") -> None:
        """Attach an interactive Plotly ``figure`` to :attr:`plots`."""
        self.plots[name] = figure

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

        call_params = dict(params or {})
        if order is not None:
            call_params.setdefault("order", order)

        target = self.clone() if clone else self
        result = engine(target.products, target.meta, call_params)
        if not isinstance(result, StepResult):
            raise TypeError(
                f"engine for {name!r} returned {type(result).__name__}, "
                "expected StepResult"
            )

        target.products.update(result.products)
        target.meta.update(result.meta)
        if qa:
            target.plots.update(result.plots)
        for masked_order, reason in result.masked_orders.items():
            target.order_mask.mask(masked_order, reason)
        # Keep the high-water mark: re-running an earlier step (e.g. a single
        # order) must not rewind the recorded progress.
        target.step = max(target.step, ordinal)
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
