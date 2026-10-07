"""Pipeline tier: state management, FITS persistence, and step orchestration."""

from onfire.pipeline.config import DEFAULT_CONFIG, load_config
from onfire.pipeline.runner import (
    FailFastError,
    STEP_ROLES,
    format_reproduction_code,
    reduce,
    run,
)
from onfire.pipeline.state import OrderMask, ReductionState
from onfire.pipeline.steps import (
    STEP_SEQUENCE,
    StepEngine,
    StepResult,
    clear_step_engines,
    get_step_comparison,
    get_step_engine,
    register_step_comparison,
    register_step_engine,
    registered_steps,
    step_name,
    step_ordinal,
)

__all__ = [
    "DEFAULT_CONFIG",
    "STEP_SEQUENCE",
    "STEP_ROLES",
    "FailFastError",
    "OrderMask",
    "ReductionState",
    "StepEngine",
    "StepResult",
    "clear_step_engines",
    "format_reproduction_code",
    "get_step_comparison",
    "get_step_engine",
    "load_config",
    "reduce",
    "register_step_comparison",
    "register_step_engine",
    "registered_steps",
    "run",
    "step_name",
    "step_ordinal",
]
