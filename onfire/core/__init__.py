"""Computational tier: pure functional Step Engines operating on NumPy arrays.

Algorithms here are decoupled from state management and persistence (ADR 0003).
Importing this package registers the implemented engines with the Step Engine
registry used by :class:`~onfire.pipeline.state.ReductionState`.
"""

from onfire.core.calibrations import (
    build_bad_pixel_mask,
    combine_flats,
    normalize_flat,
)
from onfire.core.engines import (
    prep_calibrations,
    register_engines,
    trace_orders,
)
from onfire.core.fire import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
)
from onfire.core.tracing import (
    detect_order_edges,
    evaluate_edge_trace,
    fit_edge_traces,
    trace_columns,
)

register_engines()

__all__ = [
    "FIRE_NORDERS",
    "FIRE_ORDER_NUMBERS",
    "FIRE_ORDER_SPAT_POS",
    "build_bad_pixel_mask",
    "combine_flats",
    "detect_order_edges",
    "evaluate_edge_trace",
    "fit_edge_traces",
    "normalize_flat",
    "prep_calibrations",
    "register_engines",
    "trace_columns",
    "trace_orders",
]
