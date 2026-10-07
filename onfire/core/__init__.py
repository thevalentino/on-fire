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
    find_objects,
    fit_tilts,
    prep_calibrations,
    process_science,
    register_engines,
    subtract_sky,
    trace_orders,
)
from onfire.core.fire import (
    FIRE_NORDERS,
    FIRE_ORDER_NUMBERS,
    FIRE_ORDER_SPAT_POS,
)
from onfire.core.objects import (
    ObjectTrace,
    collapse_profile,
    evaluate_trace,
    find_profile_peaks,
    fit_trace,
    profile_snr,
    rectify_order,
    trace_object,
)
from onfire.core.polyfit import PolynomialFit, robust_legendre_fit
from onfire.core.science import (
    combine_frames,
    flat_field,
    frame_variance,
    pair_difference,
)
from onfire.core.skysub import BSplineBasis, SkyFit, fit_sky_surface, tilted_coordinates
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
    "BSplineBasis",
    "FIRE_NORDERS",
    "FIRE_ORDER_NUMBERS",
    "FIRE_ORDER_SPAT_POS",
    "ObjectTrace",
    "PolynomialFit",
    "SkyFit",
    "TiltFit",
    "WavelengthSolution",
    "build_bad_pixel_mask",
    "build_tilt_fit",
    "calibrate_wavelengths",
    "collapse_profile",
    "combine_flats",
    "combine_frames",
    "detect_lines",
    "detect_order_edges",
    "evaluate_edge_trace",
    "evaluate_tilt_surface",
    "evaluate_trace",
    "evaluate_wavelength_solution",
    "evaluate_wavelength_surface",
    "find_objects",
    "find_profile_peaks",
    "fit_edge_traces",
    "fit_global_tilt_surface",
    "fit_sky_surface",
    "fit_tilt_surface",
    "fit_tilts",
    "fit_trace",
    "fit_wavelength_solution",
    "fit_wavelength_surface",
    "flat_field",
    "frame_variance",
    "normalize_flat",
    "pair_difference",
    "prep_calibrations",
    "process_science",
    "profile_snr",
    "rectify_order",
    "register_engines",
    "regularize_tilt_fit",
    "regularize_wavelength_solution",
    "reidentify_spectrum",
    "robust_legendre_fit",
    "subtract_sky",
    "tilt_angle",
    "tilted_coordinates",
    "trace_columns",
    "trace_line_tilt",
    "trace_object",
    "trace_orders",
    "trace_tilts",
    "xcorr_shift",
]
