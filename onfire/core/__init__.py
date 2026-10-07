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
    calibrate_wavelengths,
    fit_tilts,
    prep_calibrations,
    register_engines,
    trace_orders,
)
from onfire.core.fire import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
)
from onfire.core.polyfit import PolynomialFit, robust_legendre_fit
from onfire.core.tilts import (
    TiltFit,
    build_tilt_fit,
    evaluate_tilt_surface,
    fit_global_tilt_surface,
    fit_tilt_surface,
    regularize_tilt_fit,
    tilt_angle,
    trace_line_tilt,
    trace_tilts,
)
from onfire.core.tracing import (
    detect_order_edges,
    evaluate_edge_trace,
    fit_edge_traces,
    trace_columns,
)
from onfire.core.wavelengths import (
    WavelengthSolution,
    detect_lines,
    evaluate_wavelength_solution,
    evaluate_wavelength_surface,
    fit_wavelength_solution,
    fit_wavelength_surface,
    reidentify_spectrum,
    regularize_wavelength_solution,
    xcorr_shift,
)

register_engines()

__all__ = [
    "FIRE_NORDERS",
    "FIRE_ORDER_NUMBERS",
    "FIRE_ORDER_SPAT_POS",
    "PolynomialFit",
    "TiltFit",
    "WavelengthSolution",
    "build_bad_pixel_mask",
    "build_tilt_fit",
    "calibrate_wavelengths",
    "combine_flats",
    "detect_lines",
    "detect_order_edges",
    "evaluate_edge_trace",
    "evaluate_tilt_surface",
    "evaluate_wavelength_solution",
    "evaluate_wavelength_surface",
    "fit_edge_traces",
    "fit_global_tilt_surface",
    "fit_tilt_surface",
    "fit_tilts",
    "fit_wavelength_solution",
    "fit_wavelength_surface",
    "normalize_flat",
    "prep_calibrations",
    "register_engines",
    "regularize_tilt_fit",
    "regularize_wavelength_solution",
    "reidentify_spectrum",
    "robust_legendre_fit",
    "tilt_angle",
    "trace_columns",
    "trace_line_tilt",
    "trace_orders",
    "trace_tilts",
    "xcorr_shift",
]
