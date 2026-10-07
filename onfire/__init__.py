"""onfire: a streamlined reduction pipeline for Magellan/FIRE echelle data."""

from onfire import core  # noqa: F401
from onfire.pipeline import (
    FailFastError,
    ReductionState,
    format_reproduction_code,
    reduce,
    run,
)

__all__ = [
    "FailFastError",
    "ReductionState",
    "core",
    "format_reproduction_code",
    "reduce",
    "run",
]

__version__ = "0.1.0"
