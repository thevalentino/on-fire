"""Step Engine adapters for calibration prep and order edge tracing.

Each engine reads its inputs from the state products dictionary, calls the pure
functions in :mod:`onfire.core.calibrations` / :mod:`onfire.core.tracing`, and
returns a :class:`~onfire.pipeline.steps.StepResult`. No state object is ever
seen here (ADR 0003).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.table import Table
from plotly import graph_objects as go

from onfire.core.calibrations import (
    build_bad_pixel_mask,
    combine_flats,
    normalize_flat,
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
from onfire.core.polyfit import PolynomialFit
from onfire.core.science import (
    combine_frames,
    flat_field,
    frame_variance,
    pair_difference,
)
from onfire.core.skysub import fit_sky_surface, tilted_coordinates
from onfire.core.tilts import (
    TiltFit,
    build_tilt_fit,
    fit_global_tilt_surface,
    fit_tilt_surface,
    regularize_tilt_fit,
    trace_tilts,
)
from onfire.core.tracing import (
    detect_order_edges,
    evaluate_edge_trace,
    fit_edge_traces,
    trace_coordinate,
    trace_columns,
)
from onfire.core.wavelengths import (
    WavelengthSolution,
    detect_lines,
    evaluate_wavelength_surface,
    fit_wavelength_solution,
    fit_wavelength_surface,
    reidentify_spectrum,
    regularize_wavelength_solution,
)
from onfire.pipeline.steps import StepResult, register_step_engine

ORDER_TRACE_COLUMNS = (
    "order",
    "left_cen",
    "right_cen",
    "left_coeff",
    "right_coeff",
    "xmid",
    "xscale",
    "rms_left",
    "rms_right",
    "masked",
)

WAVELENGTH_COLUMNS = (
    "order",
    "nlines",
    "rms",
    "cen_wave",
    "cen_disp",
    "masked",
)

TILT_COLUMNS = (
    "order",
    "nlines",
    "rms",
    "tilt_angle",
    "masked",
)


def _target_order_index(params: dict[str, Any]) -> int | None:
    """Return the FIRE order index requested by ``params['order']``, if any."""
    target = params.get("order")
    if target is None:
        return None
    matches = np.where(FIRE_ORDER_NUMBERS == int(target))[0]
    if matches.size == 0:
        raise ValueError(f"order {target} is not a FIRE echelle order 11-31")
    return int(matches[0])


def _tilt_prior_from_coeffs(
    coeffs: np.ndarray | None, spec_order: int, spat_order: int
) -> PolynomialFit:
    """Build a global tilt prior surface from stored coefficients."""
    nterms = spec_order * spat_order
    if coeffs is None:
        array = np.zeros(nterms)
    else:
        array = np.asarray(coeffs, dtype=float)
        if array.size < nterms:
            array = np.concatenate([array, np.zeros(nterms - array.size)])
        array = array[:nterms]
        if not np.all(np.isfinite(array)):
            array = np.zeros(nterms)
    return PolynomialFit(array, (max(0, spec_order - 1), max(0, spat_order - 1)), [(0.0, 1.0), (-1.0, 1.0)])


def prep_calibrations(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Combine raw flats and build a detector bad-pixel mask (reduction step 1)."""
    raw = products.get("raw_flats")
    if raw is None:
        raise ValueError(
            "prep_calibrations requires a 'raw_flats' product; "
            "load raw flat exposures first"
        )
    stack = np.asarray(raw, dtype=float)
    combined, variance, good, nused = combine_flats(
        stack,
        sigma_clip=bool(params.get("sigma_clip", True)),
        sigrej=params.get("sigrej"),
    )
    norm = normalize_flat(combined, window=int(params.get("normalize_window", 101)))
    bad = build_bad_pixel_mask(
        combined,
        n_sigma=float(params.get("bad_pixel_sigma", 5.0)),
        window=int(params.get("bad_pixel_window", 5)),
    )

    figure = go.Figure(
        data=go.Heatmap(z=norm, colorscale="Viridis", colorbar={"title": "norm"})
    )
    figure.update_layout(title="Master flat", xaxis_title="spatial", yaxis_title="dispersion")

    return StepResult(
        products={
            "flat": combined,
            "flat_var": variance,
            "flat_norm": norm,
            "bad_pixel_mask": bad,
        },
        meta={
            "n_flats": int(stack.shape[0]),
            "n_bad_pixels": int(bad.sum()),
            "flat_good": bool(good.all()),
            "flat_nused_median": float(np.median(nused)),
        },
        plots={"flat": figure},
    )


def _blank_trace_table(ncoeff: int) -> Table:
    return Table(
        {
            "order": np.zeros(FIRE_NORDERS, dtype=np.int16),
            "left_cen": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "right_cen": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "left_coeff": np.full((FIRE_NORDERS, ncoeff), np.nan, dtype=np.float64),
            "right_coeff": np.full((FIRE_NORDERS, ncoeff), np.nan, dtype=np.float64),
            "xmid": np.zeros(FIRE_NORDERS, dtype=np.float32),
            "xscale": np.ones(FIRE_NORDERS, dtype=np.float32),
            "rms_left": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "rms_right": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "masked": np.ones(FIRE_NORDERS, dtype=bool),
        },
        names=ORDER_TRACE_COLUMNS,
    )


def _existing_entries(
    table: Table,
) -> dict[int, tuple[np.ndarray, np.ndarray, float, float]]:
    existing: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for row in table:
        existing[int(row["order"])] = (
            np.asarray(row["left_coeff"], dtype=float),
            np.asarray(row["right_coeff"], dtype=float),
            float(row["rms_left"]),
            float(row["rms_right"]),
        )
    return existing


def trace_orders(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Trace the left and right spatial edges of orders 11-31 (reduction step 2).

    When ``params['order']`` targets a single order and an existing
    ``order_traces`` product is present, only that order is re-traced; all other
    orders keep their previous fits.
    """
    flat = products.get("flat_norm")
    if flat is None:
        flat = products.get("flat")
    if flat is None:
        raise ValueError(
            "trace_orders requires a 'flat_norm' or 'flat' product; "
            "run prep_calibrations first"
        )
    flat = np.asarray(flat, dtype=float)
    nspec = flat.shape[0]
    ncoeff = int(params.get("ncoeff", 5))
    maxshift = float(params.get("maxshift", 3.0))
    ref_row = (nspec - 1) // 2
    xmid, xscale = trace_coordinate(nspec)

    left0, right0, _ = detect_order_edges(flat, FIRE_ORDER_SPAT_POS)
    gradient = np.gradient(flat, axis=1)

    existing_table = products.get("order_traces")
    target = params.get("order")
    target_index: int | None = None
    if target is not None:
        matches = np.where(FIRE_ORDER_NUMBERS == int(target))[0]
        if matches.size == 0:
            raise ValueError(f"order {target} is not a FIRE echelle order 11-31")
        target_index = int(matches[0])

    coefficients: list[np.ndarray | None] = [None] * (2 * FIRE_NORDERS)
    edge_rms: list[float] = [float("nan")] * (2 * FIRE_NORDERS)
    reusing = target_index is not None and existing_table is not None
    if reusing:
        for order, (left_coeff, right_coeff, left_rms, right_rms) in _existing_entries(
            existing_table
        ).items():
            index = int(np.where(FIRE_ORDER_NUMBERS == order)[0][0])
            coefficients[2 * index] = left_coeff
            coefficients[2 * index + 1] = right_coeff
            edge_rms[2 * index] = left_rms
            edge_rms[2 * index + 1] = right_rms

    indices = [target_index] if reusing else list(range(FIRE_NORDERS))
    starts: list[float] = []
    signs: list[float] = []
    slot_of_local: list[int] = []
    for index in indices:
        for side, seed in ((0, left0[index]), (1, right0[index])):
            slot_of_local.append(2 * index + side)
            starts.append(float(seed))
            signs.append(1.0 if side == 0 else -1.0)

    tracked = trace_columns(
        gradient, np.asarray(starts), np.asarray(signs), maxshift=maxshift, ref_row=ref_row
    )
    fitted, rms = fit_edge_traces(tracked, ncoeff=ncoeff, ref_row=ref_row)
    for local, slot in enumerate(slot_of_local):
        coefficients[slot] = fitted[local]
        edge_rms[slot] = rms[local]

    table = _blank_trace_table(ncoeff)
    edge_image = np.full((nspec, 2 * FIRE_NORDERS), np.nan, dtype=float)
    rows = np.arange(nspec)
    masked_orders: dict[int, str] = {}
    for index, order in enumerate(FIRE_ORDER_NUMBERS):
        left_coeff = coefficients[2 * index]
        right_coeff = coefficients[2 * index + 1]
        table["order"][index] = order
        table["xmid"][index] = xmid
        table["xscale"][index] = xscale
        if left_coeff is None or right_coeff is None or not (
            np.all(np.isfinite(left_coeff)) and np.all(np.isfinite(right_coeff))
        ):
            table["masked"][index] = True
            masked_orders[int(order)] = "edge trace fit failed"
            continue
        left_positions = evaluate_edge_trace(left_coeff, xmid, xscale, rows)
        right_positions = evaluate_edge_trace(right_coeff, xmid, xscale, rows)
        edge_image[:, 2 * index] = left_positions
        edge_image[:, 2 * index + 1] = right_positions
        table["left_coeff"][index] = left_coeff
        table["right_coeff"][index] = right_coeff
        table["left_cen"][index] = left_positions[ref_row]
        table["right_cen"][index] = right_positions[ref_row]
        table["rms_left"][index] = edge_rms[2 * index]
        table["rms_right"][index] = edge_rms[2 * index + 1]
        table["masked"][index] = False

    figure = _edge_figure(flat, edge_image)
    n_masked = int(table["masked"].sum())
    return StepResult(
        products={"order_traces": table, "edge_traces": edge_image},
        meta={
            "n_orders": FIRE_NORDERS,
            "n_orders_traced": int(FIRE_NORDERS - n_masked),
            "n_orders_masked": n_masked,
            "ncoeff": ncoeff,
        },
        plots={"edge_traces": figure},
        masked_orders=masked_orders,
    )


def _edge_figure(flat: np.ndarray, edge_image: np.ndarray) -> go.Figure:
    nspec = flat.shape[0]
    stride = max(1, nspec // 100)
    rows = np.arange(0, nspec, stride)
    figure = go.Figure(
        data=go.Heatmap(
            z=flat,
            colorscale="Gray",
            colorbar={"title": "counts"},
            opacity=1.0,
        )
    )
    for column in range(edge_image.shape[1]):
        positions = edge_image[rows, column]
        valid = np.isfinite(positions)
        side = "left" if column % 2 == 0 else "right"
        order = FIRE_ORDER_NUMBERS[column // 2]
        figure.add_trace(
            go.Scatter(
                x=positions[valid],
                y=rows[valid],
                mode="markers",
                marker={"size": 2, "color": "red" if side == "left" else "cyan"},
                name=f"order {order} {side}",
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Order edge traces",
        xaxis_title="spatial",
        yaxis_title="dispersion",
    )
    return figure


def _combine_arcs(products: dict[str, Any]) -> np.ndarray:
    raw = products.get("raw_arcs")
    if raw is not None:
        stack = np.asarray(raw, dtype=float)
        if stack.ndim == 2:
            stack = stack[np.newaxis, :, :]
        return np.nanmean(stack, axis=0)
    image = products.get("arc_image")
    if image is None:
        raise ValueError(
            "calibrate_wavelengths requires a 'raw_arcs' or 'arc_image' product; "
            "load raw arc exposures first"
        )
    return np.asarray(image, dtype=float)


def _trace_edges(
    traces: Table, order: int
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    match = traces[traces["order"] == order]
    if len(match) == 0:
        return None
    row = match[0]
    if bool(row["masked"]):
        return None
    left = np.asarray(row["left_coeff"], dtype=float)
    right = np.asarray(row["right_coeff"], dtype=float)
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return None
    return left, right, float(row["xmid"]), float(row["xscale"])


def _extract_order_spectra(
    arcimg: np.ndarray, traces: Table, box_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nspec, nspat = arcimg.shape
    spectra = np.full((nspec, FIRE_NORDERS), np.nan)
    centers = np.full((nspec, FIRE_NORDERS), np.nan)
    widths = np.full(FIRE_NORDERS, np.nan)
    rows = np.arange(nspec)
    for index, order in enumerate(FIRE_ORDER_NUMBERS):
        edges = _trace_edges(traces, int(order))
        if edges is None:
            continue
        left_coeff, right_coeff, xmid, xscale = edges
        left = evaluate_edge_trace(left_coeff, xmid, xscale, rows)
        right = evaluate_edge_trace(right_coeff, xmid, xscale, rows)
        center = 0.5 * (left + right)
        centers[:, index] = center
        widths[index] = float(np.nanmedian(0.5 * (right - left)))
        spectrum = np.zeros(nspec)
        for row in range(nspec):
            lo = max(0, int(round(center[row] - box_rad)))
            hi = min(nspat, int(round(center[row] + box_rad)) + 1)
            if hi > lo:
                spectrum[row] = float(np.nansum(arcimg[row, lo:hi]))
        spectra[:, index] = spectrum
    return spectra, centers, widths


def _blank_wavelength_table() -> Table:
    return Table(
        {
            "order": FIRE_ORDER_NUMBERS.astype(np.int16),
            "nlines": np.zeros(FIRE_NORDERS, dtype=np.int32),
            "rms": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "cen_wave": np.full(FIRE_NORDERS, np.nan, dtype=np.float64),
            "cen_disp": np.full(FIRE_NORDERS, np.nan, dtype=np.float64),
            "masked": np.ones(FIRE_NORDERS, dtype=bool),
        },
        names=WAVELENGTH_COLUMNS,
    )


def _write_wavelength_row(
    table: Table,
    index: int,
    solution: WavelengthSolution,
) -> None:
    table["nlines"][index] = solution.nlines
    table["rms"][index] = solution.rms
    table["cen_wave"][index] = solution.cen_wave
    table["cen_disp"][index] = solution.cen_disp
    table["masked"][index] = solution.masked


def _evaluate_wavelength_surface(
    fit: PolynomialFit, nspec: int, order_min: int, order_max: int
) -> np.ndarray:
    rows = np.arange(nspec, dtype=float)
    surface = np.full((nspec, FIRE_NORDERS), np.nan)
    for index, order in enumerate(FIRE_ORDER_NUMBERS):
        surface[:, index] = evaluate_wavelength_surface(
            fit,
            rows,
            np.full(nspec, int(order), dtype=float),
            nspec=nspec,
            order_min=order_min,
            order_max=order_max,
        )
    return surface


def _wavelength_residual_figure(
    table: Table, wave_soln: np.ndarray, line_ids: Table
) -> go.Figure:
    figure = go.Figure()
    rows = np.arange(wave_soln.shape[0], dtype=float)
    for index, order in enumerate(FIRE_ORDER_NUMBERS):
        selected = line_ids[line_ids["order"] == order]
        if len(selected) == 0 or not np.all(np.isfinite(wave_soln[:, index])):
            continue
        pixel = np.asarray(selected["pixel"], dtype=float)
        wave = np.asarray(selected["wave"], dtype=float)
        model = np.interp(pixel, rows, wave_soln[:, index])
        dispersion = abs(float(np.median(np.diff(wave_soln[:, index])))) or 1.0
        figure.add_trace(
            go.Scatter(
                x=pixel,
                y=(wave - model) / dispersion,
                mode="markers",
                name=f"order {int(order)}",
            )
        )
    figure.update_layout(
        title="Wavelength fit residuals",
        xaxis_title="dispersion pixel",
        yaxis_title="residual (pixels)",
    )
    return figure


def _wavelength_identification_figure(
    spectra: np.ndarray,
    table: Table,
    line_ids: Table,
    wave_soln: np.ndarray,
) -> go.Figure:
    figure = go.Figure()
    unmasked = np.where(~np.asarray(table["masked"], dtype=bool))[0]
    counts = [int((line_ids["order"] == FIRE_ORDER_NUMBERS[i]).sum()) for i in range(FIRE_NORDERS)]
    index = max(unmasked, key=lambda i: counts[i]) if unmasked.size else 0
    order = int(FIRE_ORDER_NUMBERS[index])
    pixel = np.arange(spectra.shape[0])
    figure.add_trace(
        go.Scatter(
            x=pixel,
            y=spectra[:, index],
            mode="lines",
            name=f"order {order} arc",
        )
    )
    selected = line_ids[line_ids["order"] == order]
    if len(selected):
        line_pixel = np.asarray(selected["pixel"], dtype=float)
        wave = np.asarray(selected["wave"], dtype=float)
        amplitude = np.interp(line_pixel, pixel, spectra[:, index])
        figure.add_trace(
            go.Scatter(
                x=line_pixel,
                y=amplitude,
                mode="markers+text",
                text=[f"{w:.1f}" for w in wave],
                textposition="top center",
                marker={"color": "red", "size": 5},
                name="identifications",
            )
        )
    figure.update_layout(
        title=f"Arc line identifications (order {order})",
        xaxis_title="dispersion pixel",
        yaxis_title="counts",
    )
    return figure


def calibrate_wavelengths(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Identify arc/OH lines and fit 1D dispersion solutions (reduction step 3).

    In a full run every order is reidentified against the bundled FIRE archive
    and a global 2D pixel-vs-order surface is fit across the orders. When
    ``params['order']`` targets a single order and a previous solution exists,
    that order is refined with cross-order regularization from the global
    surface while all other orders keep their fits.
    """
    from onfire.data import load_fire_line_list, load_fire_template

    arcimg = _combine_arcs(products)
    traces = products.get("order_traces")
    if traces is None:
        raise ValueError(
            "calibrate_wavelengths requires an 'order_traces' product; "
            "run trace_orders first"
        )
    nspec = arcimg.shape[0]
    ncoeff = int(params.get("ncoeff", 5))
    box_rad = float(params.get("arc_box_rad", 3.0))
    sigdetect = float(params.get("sigdetect", 5.0))
    fwhm = float(params.get("fwhm", 4.0))
    match_toler = float(params.get("match_toler", 2.0))
    n_first = int(params.get("n_first", 2))
    n_final = int(params.get("n_final", 4))
    nspec_coeff = int(params.get("nspec_coeff", 4))
    norder_coeff = int(params.get("norder_coeff", 6))

    template = load_fire_template(params.get("template_path"))
    line_list = load_fire_line_list(params.get("line_list_path"))
    line_waves = np.asarray(line_list["wave"], dtype=float)
    template_orders = np.asarray(template["order"], dtype=int)
    template_spectra = {
        int(order): (
            np.asarray(template["wave"][i], dtype=float),
            np.abs(np.asarray(template["flux"][i], dtype=float)),
        )
        for i, order in enumerate(template_orders)
    }

    spectra, centers, widths = _extract_order_spectra(arcimg, traces, box_rad)
    order_min = int(FIRE_ORDER_NUMBERS.min())
    order_max = int(FIRE_ORDER_NUMBERS.max())

    target_index = _target_order_index(params)
    previous_table = products.get("wavelengths")
    previous_soln = products.get("wave_soln")
    regen = target_index is not None and previous_soln is not None

    wave_soln = np.full((nspec, FIRE_NORDERS), np.nan)
    wave_coeffs = np.full((FIRE_NORDERS, ncoeff), np.nan)
    table = _blank_wavelength_table()
    records: list[tuple[int, float, float]] = []
    rows = np.arange(nspec, dtype=float)
    masked_orders: dict[int, str] = {}

    if regen:
        wave_soln[:] = np.asarray(previous_soln, dtype=float)
        if previous_table is not None:
            for name in WAVELENGTH_COLUMNS:
                table[name] = previous_table[name]
        if products.get("wave_coeffs") is not None:
            wave_coeffs[:] = np.asarray(products["wave_coeffs"], dtype=float)
        prior_surface = products.get("wave_surface")
        if prior_surface is not None:
            prior = np.asarray(prior_surface, dtype=float)[:, target_index]
        else:
            prior = wave_soln[:, target_index]
        solution = regularize_wavelength_solution(
            np.where(np.isfinite(spectra[:, target_index]), spectra[:, target_index], 0.0),
            line_waves,
            prior,
            sigdetect=sigdetect,
            fwhm=fwhm,
            match_toler=max(match_toler, 3.0),
            ncoeff=ncoeff,
        )
        wave_soln[:, target_index] = solution.wave_soln
        coeff, _, _ = fit_wavelength_solution(
            rows, solution.wave_soln, ncoeff=ncoeff, nspec=nspec
        )
        wave_coeffs[target_index] = coeff
        _write_wavelength_row(table, target_index, solution)
        records.extend(
            (int(FIRE_ORDER_NUMBERS[target_index]), float(p), float(w))
            for p, w in zip(solution.pixel_fit, solution.wave_fit)
        )
        wave_surface = prior_surface
        stored_coeffs = products.get("wave_surface_coeffs")
        if stored_coeffs is None:
            wave_surface_coeffs = np.full(nspec_coeff * norder_coeff, np.nan)
        else:
            wave_surface_coeffs = np.asarray(stored_coeffs, dtype=float)
    else:
        matched_pixel: list[float] = []
        matched_wave: list[float] = []
        matched_order: list[int] = []
        for index, order in enumerate(FIRE_ORDER_NUMBERS):
            order = int(order)
            if order not in template_spectra or not np.all(np.isfinite(spectra[:, index])):
                table["masked"][index] = True
                masked_orders[order] = "no arc spectrum"
                continue
            arxiv_wave, arxiv_flux = template_spectra[order]
            solution = reidentify_spectrum(
                spectra[:, index],
                arxiv_flux,
                arxiv_wave,
                line_waves,
                sigdetect=sigdetect,
                fwhm=fwhm,
                match_toler=match_toler,
                n_first=n_first,
                n_final=n_final,
            )
            if solution.masked:
                table["masked"][index] = True
                masked_orders[order] = "insufficient line identifications"
                continue
            wave_soln[:, index] = solution.wave_soln
            coeff, _, _ = fit_wavelength_solution(
                rows, solution.wave_soln, ncoeff=ncoeff, nspec=nspec
            )
            wave_coeffs[index] = coeff
            _write_wavelength_row(table, index, solution)
            records.extend(
                (order, float(p), float(w))
                for p, w in zip(solution.pixel_fit, solution.wave_fit)
            )
            matched_pixel.extend(solution.pixel_fit.tolist())
            matched_wave.extend(solution.wave_fit.tolist())
            matched_order.extend([order] * solution.pixel_fit.size)

        if len(matched_pixel) >= nspec_coeff * norder_coeff:
            fit, _ = fit_wavelength_surface(
                np.asarray(matched_pixel, dtype=float),
                np.asarray(matched_wave, dtype=float),
                np.asarray(matched_order, dtype=float),
                nspec=nspec,
                order_min=order_min,
                order_max=order_max,
                nspec_coeff=nspec_coeff,
                norder_coeff=norder_coeff,
            )
            wave_surface_coeffs = np.asarray(fit.coeffs, dtype=float)
            wave_surface = _evaluate_wavelength_surface(
                fit, nspec, order_min, order_max
            )
        else:
            wave_surface_coeffs = np.full(nspec_coeff * norder_coeff, np.nan)
            wave_surface = wave_soln.copy()

    line_ids = Table(
        {
            "order": np.asarray([r[0] for r in records], dtype=np.int16),
            "pixel": np.asarray([r[1] for r in records], dtype=np.float64),
            "wave": np.asarray([r[2] for r in records], dtype=np.float64),
        },
        names=("order", "pixel", "wave"),
    )
    if wave_surface is None:
        wave_surface = wave_soln.copy()

    residuals = _wavelength_residual_figure(table, wave_soln, line_ids)
    identifications = _wavelength_identification_figure(
        spectra, table, line_ids, wave_soln
    )

    n_masked = int(np.asarray(table["masked"], dtype=bool).sum())
    return StepResult(
        products={
            "arc_image": arcimg,
            "arc_spectra": spectra,
            "order_centers": centers,
            "order_halfwidths": widths,
            "wavelengths": table,
            "wave_soln": wave_soln,
            "wave_coeffs": wave_coeffs,
            "wave_surface": np.asarray(wave_surface, dtype=float),
            "wave_surface_coeffs": wave_surface_coeffs,
            "line_ids": line_ids,
        },
        meta={
            "n_orders_calibrated": int(FIRE_NORDERS - n_masked),
            "n_orders_masked": n_masked,
            "n_lines": int(line_ids["order"].size),
            "wave_surface_nspec_coeff": nspec_coeff,
            "wave_surface_norder_coeff": norder_coeff,
            "wave_order_min": order_min,
            "wave_order_max": order_max,
        },
        plots={
            "wavelength_residuals": residuals,
            "wavelength_identifications": identifications,
        },
        masked_orders=masked_orders,
    )


def _blank_tilt_table() -> Table:
    return Table(
        {
            "order": FIRE_ORDER_NUMBERS.astype(np.int16),
            "nlines": np.zeros(FIRE_NORDERS, dtype=np.int32),
            "rms": np.full(FIRE_NORDERS, np.nan, dtype=np.float32),
            "tilt_angle": np.full(FIRE_NORDERS, np.nan, dtype=np.float64),
            "masked": np.ones(FIRE_NORDERS, dtype=bool),
        },
        names=TILT_COLUMNS,
    )


def _write_tilt_row(table: Table, index: int, fit: TiltFit) -> None:
    table["nlines"][index] = fit.nlines
    table["rms"][index] = fit.rms
    table["tilt_angle"][index] = fit.tilt_angle
    table["masked"][index] = fit.masked


def _tilt_residual_figure(measurements: Table, fits: list[TiltFit]) -> go.Figure:
    figure = go.Figure()
    for index, fit in enumerate(fits):
        if fit is None or fit.masked:
            continue
        order = int(FIRE_ORDER_NUMBERS[index])
        selected = measurements[measurements["order"] == order]
        if len(selected) == 0:
            continue
        residual = np.asarray(selected["delta"], dtype=float) - np.asarray(
            selected["model"], dtype=float
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(selected["spec_pixel"], dtype=float),
                y=residual,
                mode="markers",
                marker={"size": 3},
                name=f"order {order}",
            )
        )
    figure.update_layout(
        title="Tilt surface residuals",
        xaxis_title="dispersion pixel",
        yaxis_title="residual (pixels)",
    )
    return figure


def _tilt_map_figure(
    arcimg: np.ndarray,
    fits: list[TiltFit | None],
    spat_scale: float,
) -> go.Figure:
    valid = [i for i, fit in enumerate(fits) if fit is not None and not fit.masked]
    index = valid[0] if valid else 0
    order = int(FIRE_ORDER_NUMBERS[index])
    nspec = arcimg.shape[0]
    dspat = np.linspace(-spat_scale, spat_scale, 41)
    pixel = np.arange(nspec, dtype=float)
    pixel_grid, dspat_grid = np.meshgrid(pixel, dspat, indexing="ij")
    fit = fits[index]
    model = np.full_like(pixel_grid, np.nan)
    if fit is not None and not fit.masked:
        poly = build_tilt_fit(fit)
        model = poly.evaluate([pixel_grid / fit.spec_scale, dspat_grid / fit.spat_scale])
        model = model * fit.spec_scale
    figure = go.Figure(
        data=go.Heatmap(
            x=dspat,
            y=pixel,
            z=model,
            colorscale="RdBu",
            colorbar={"title": "delta"},
        )
    )
    figure.update_layout(
        title=f"Tilt model (order {order})",
        xaxis_title="spatial offset (pixels)",
        yaxis_title="dispersion pixel",
    )
    return figure


def fit_tilts(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Model 2D spectral line tilts across each order (reduction step 4).

    Arc lines are traced across the spatial direction of every order and a 2D
    Legendre surface of the spectral-position offset is fit. A global surface
    pooled across all orders provides cross-order regularization when
    ``params['order']`` targets a single order.
    """
    arcimg = products.get("arc_image")
    spectra = products.get("arc_spectra")
    centers = products.get("order_centers")
    widths = products.get("order_halfwidths")
    if arcimg is None or spectra is None or centers is None:
        raise ValueError(
            "fit_tilts requires 'arc_image', 'arc_spectra' and 'order_centers' "
            "products; run calibrate_wavelengths first"
        )
    nspec = arcimg.shape[0]
    spec_order = int(params.get("spec_order", 3))
    spat_order = int(params.get("spat_order", 4))
    sigdetect = float(params.get("tilt_sigdetect", 10.0))
    fwhm = float(params.get("fwhm", 4.0))
    maxshift = float(params.get("tilt_maxshift", 3.0))
    nterms = spec_order * spat_order
    finite_widths = np.asarray(widths, dtype=float)[np.isfinite(widths)]
    spat_scale = float(
        params.get("spat_scale", np.median(finite_widths) - 2.0 if finite_widths.size else 10.0)
    )
    spat_scale = max(spat_scale, 1.0)

    previous_coeffs = products.get("tilt_coeffs")
    previous_table = products.get("tilts")
    target_index = _target_order_index(params)
    regen = target_index is not None and previous_coeffs is not None

    tilt_coeffs = np.full((FIRE_NORDERS, nterms), np.nan)
    table = _blank_tilt_table()
    if regen:
        tilt_coeffs[:] = np.asarray(previous_coeffs, dtype=float)
        if previous_table is not None:
            for name in TILT_COLUMNS:
                table[name] = previous_table[name]
        prior = _tilt_prior_from_coeffs(
            products.get("tilt_surface_coeffs"), spec_order, spat_order
        )
    else:
        prior = None

    indices = [target_index] if regen else list(range(FIRE_NORDERS))
    fits: list[TiltFit | None] = [None] * FIRE_NORDERS
    measurement_records: list[tuple[int, float, float, float, float]] = []
    pooled_spec: list[float] = []
    pooled_dspat: list[float] = []
    pooled_delta: list[float] = []
    masked_orders: dict[int, str] = {}

    for index in indices:
        order = int(FIRE_ORDER_NUMBERS[index])
        if (
            not np.all(np.isfinite(spectra[:, index]))
            or not np.any(np.isfinite(centers[:, index]))
        ):
            table["masked"][index] = True
            masked_orders[order] = "no arc spectrum"
            continue
        detected = detect_lines(spectra[:, index], sigdetect=sigdetect, fwhm=fwhm)
        if detected.tcent.size < 3:
            table["masked"][index] = True
            masked_orders[order] = "too few arc lines for tilts"
            continue
        center_spat = float(np.nanmedian(centers[:, index]))
        half_width = max(2.0, float(widths[index]) - 2.0)
        measurements = trace_tilts(
            arcimg,
            detected.tcent,
            center_spat,
            half_width=half_width,
            fwhm=fwhm,
            maxshift=maxshift,
        )
        if measurements.npoints < 4:
            table["masked"][index] = True
            masked_orders[order] = "tilt tracing failed"
            continue
        if regen:
            fit = regularize_tilt_fit(
                measurements,
                prior,
                nspec=nspec,
                spat_scale=spat_scale,
                spec_order=spec_order,
                spat_order=spat_order,
            )
        else:
            fit = fit_tilt_surface(
                measurements,
                nspec=nspec,
                spat_scale=spat_scale,
                spec_order=spec_order,
                spat_order=spat_order,
            )
            pooled_spec.extend(measurements.spec_pixel.tolist())
            pooled_dspat.extend(measurements.dspat.tolist())
            pooled_delta.extend(measurements.delta.tolist())
        poly = build_tilt_fit(fit)
        model = poly.evaluate(
            [measurements.spec_pixel / fit.spec_scale, measurements.dspat / fit.spat_scale]
        ) * fit.spec_scale
        measurement_records.extend(
            (order, float(s), float(d), float(dl), float(m))
            for s, d, dl, m in zip(
                measurements.spec_pixel,
                measurements.dspat,
                measurements.delta,
                model,
            )
        )
        fits[index] = fit
        tilt_coeffs[index] = fit.coeffs
        _write_tilt_row(table, index, fit)

    if regen:
        global_coeffs = products.get("tilt_surface_coeffs")
        if global_coeffs is None:
            global_coeffs = np.full(nterms, np.nan)
        global_coeffs = np.asarray(global_coeffs, dtype=float)
    else:
        global_fit = fit_global_tilt_surface(
            np.asarray(pooled_spec, dtype=float),
            np.asarray(pooled_dspat, dtype=float),
            np.asarray(pooled_delta, dtype=float),
            nspec=nspec,
            spat_scale=spat_scale,
            spec_order=spec_order,
            spat_order=spat_order,
        )
        global_coeffs = np.asarray(global_fit.coeffs, dtype=float)

    measurements_table = Table(
        {
            "order": np.asarray([r[0] for r in measurement_records], dtype=np.int16),
            "spec_pixel": np.asarray([r[1] for r in measurement_records], dtype=np.float64),
            "dspat": np.asarray([r[2] for r in measurement_records], dtype=np.float64),
            "delta": np.asarray([r[3] for r in measurement_records], dtype=np.float64),
            "model": np.asarray([r[4] for r in measurement_records], dtype=np.float64),
        },
        names=("order", "spec_pixel", "dspat", "delta", "model"),
    )

    residuals = _tilt_residual_figure(measurements_table, fits)
    tilt_map = _tilt_map_figure(arcimg, fits, spat_scale)

    n_masked = int(np.asarray(table["masked"], dtype=bool).sum())
    return StepResult(
        products={
            "tilts": table,
            "tilt_coeffs": tilt_coeffs,
            "tilt_surface_coeffs": global_coeffs,
            "tilt_measurements": measurements_table,
        },
        meta={
            "tilt_spec_order": spec_order,
            "tilt_spat_order": spat_order,
            "tilt_spat_scale": spat_scale,
            "n_orders_tilted": int(FIRE_NORDERS - n_masked),
            "n_orders_masked": n_masked,
        },
        plots={"tilt_residuals": residuals, "tilt_map": tilt_map},
        masked_orders=masked_orders,
    )


def _simple_heatmap(z: np.ndarray, title: str, colorbar: str = "counts") -> go.Figure:
    """Build a plain detector-image heatmap QA figure."""
    figure = go.Figure(
        data=go.Heatmap(z=np.asarray(z, dtype=float), colorscale="Viridis", colorbar={"title": colorbar})
    )
    figure.update_layout(title=title, xaxis_title="spatial", yaxis_title="dispersion")
    return figure


def process_science(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Flat-field science frames and prepare background subtraction (step 5).

    The Parameter Config ``mode`` selects between Pair Differencing (``A - B``
    of consecutive nodded exposures, which cancels the thermal and OH sky) and
    Stare Mode (a single fixed pointing, whose sky is modeled downstream). The
    output carries a variance array propagated from detector noise.
    """
    raw = products.get("raw_science")
    if raw is None:
        raise ValueError(
            "process_science requires a 'raw_science' product; "
            "load raw science exposures first"
        )
    stack = np.asarray(raw, dtype=float)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]
    if stack.ndim != 3:
        raise ValueError(
            f"raw_science must have shape (nframes, nspec, nspat), got {stack.shape}"
        )

    flat = products.get("flat_norm")
    if flat is None:
        flat = products.get("flat")
    if flat is None:
        raise ValueError(
            "process_science requires a 'flat_norm' or 'flat' product; "
            "run prep_calibrations first"
        )
    flat = np.asarray(flat, dtype=float)

    gain = float(params.get("gain", 1.0))
    read_noise = float(params.get("read_noise", 0.0))
    mode = str(params.get("mode", "pair_difference")).lower()
    nframes = stack.shape[0]

    if mode == "pair_difference":
        if nframes < 2:
            raise ValueError(
                "pair differencing requires at least two science frames; "
                f"got {nframes}"
            )
        npairs = nframes // 2
        differences = np.empty((npairs, *stack.shape[1:]), dtype=float)
        variances = np.empty_like(differences)
        for pair in range(npairs):
            a = stack[2 * pair]
            b = stack[2 * pair + 1]
            differences[pair] = pair_difference(
                flat_field(a, flat), flat_field(b, flat)
            )
            variances[pair] = frame_variance(
                a, flat, gain=gain, read_noise=read_noise
            ) + frame_variance(b, flat, gain=gain, read_noise=read_noise)
        science = combine_frames(differences)
        science_var = np.nanmean(variances, axis=0) / max(1, npairs)
        ncombined = npairs
    elif mode == "stare":
        flat_frames = np.stack([flat_field(frame, flat) for frame in stack])
        science = combine_frames(flat_frames)
        frame_vars = np.stack(
            [
                frame_variance(frame, flat, gain=gain, read_noise=read_noise)
                for frame in stack
            ]
        )
        science_var = np.nanmean(frame_vars, axis=0) / max(1, nframes)
        ncombined = nframes
    else:
        raise ValueError(
            f"unknown science mode {mode!r}; expected 'pair_difference' or 'stare'"
        )

    science = np.asarray(science, dtype=float)
    science_var = np.asarray(science_var, dtype=float)
    good = np.isfinite(science) & np.isfinite(science_var) & (science_var > 0)
    bad = products.get("bad_pixel_mask")
    if bad is not None:
        good &= ~np.asarray(bad, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        ivar = np.where(
            good, 1.0 / np.maximum(science_var, 1e-12), 0.0
        )

    figure = _simple_heatmap(science, f"Science frame ({mode})")
    return StepResult(
        products={
            "science": science,
            "science_var": science_var,
            "science_ivar": ivar,
            "science_mask": good,
        },
        meta={
            "science_mode": mode,
            "n_science_frames": int(nframes),
            "n_science_combined": int(ncombined),
            "n_science_bad_pixels": int((~good).sum()),
        },
        plots={"science": figure},
    )


def _order_edges(
    traces: Table, order: int, nspec: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate one order's left/right edge positions from its stored trace."""
    match = traces[traces["order"] == order]
    if len(match) == 0:
        return None
    row = match[0]
    if bool(row["masked"]):
        return None
    left_coeff = np.asarray(row["left_coeff"], dtype=float)
    right_coeff = np.asarray(row["right_coeff"], dtype=float)
    if not (np.all(np.isfinite(left_coeff)) and np.all(np.isfinite(right_coeff))):
        return None
    xmid = float(row["xmid"])
    xscale = float(row["xscale"])
    rows = np.arange(nspec)
    left = evaluate_edge_trace(left_coeff, xmid, xscale, rows)
    right = evaluate_edge_trace(right_coeff, xmid, xscale, rows)
    return left, right


OBJECT_COLUMNS = (
    "order",
    "obj_id",
    "fracpos",
    "spat_pixpos",
    "fwhm",
    "box_radius",
    "snr",
    "rms",
    "xmid",
    "xscale",
    "ncoeff",
    "trace_coeff",
    "masked",
)


def _objects_table(traces: list[ObjectTrace], ncoeff: int) -> Table:
    table = Table(
        {
            "order": np.asarray([t.order for t in traces], dtype=np.int16),
            "obj_id": np.arange(len(traces), dtype=np.int32),
            "fracpos": np.asarray([t.fracpos for t in traces], dtype=np.float32),
            "spat_pixpos": np.asarray([t.spat_pixpos for t in traces], dtype=np.float32),
            "fwhm": np.asarray([t.fwhm for t in traces], dtype=np.float32),
            "box_radius": np.asarray([t.box_radius for t in traces], dtype=np.float32),
            "snr": np.asarray([t.snr for t in traces], dtype=np.float32),
            "rms": np.asarray([t.rms for t in traces], dtype=np.float32),
            "xmid": np.asarray([t.xmid for t in traces], dtype=np.float32),
            "xscale": np.asarray([t.xscale for t in traces], dtype=np.float32),
            "ncoeff": np.full(len(traces), ncoeff, dtype=np.int32),
            "trace_coeff": np.asarray(
                [np.asarray(t.coeff, dtype=float) for t in traces], dtype=float
            ).reshape(len(traces), ncoeff),
            "masked": np.asarray([t.masked for t in traces], dtype=bool),
        },
        names=OBJECT_COLUMNS,
    )
    return table


def _objects_from_table(table: Table) -> dict[int, list[ObjectTrace]]:
    grouped: dict[int, list[ObjectTrace]] = {}
    for row in table:
        trace = ObjectTrace(
            order=int(row["order"]),
            coeff=np.asarray(row["trace_coeff"], dtype=float),
            xmid=float(row["xmid"]),
            xscale=float(row["xscale"]),
            rms=float(row["rms"]),
            fracpos=float(row["fracpos"]),
            spat_pixpos=float(row["spat_pixpos"]),
            fwhm=float(row["fwhm"]),
            box_radius=float(row["box_radius"]),
            snr=float(row["snr"]),
            masked=bool(row["masked"]),
        )
        grouped.setdefault(trace.order, []).append(trace)
    return grouped


def _detect_order_objects(
    image: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray | None,
    left: np.ndarray,
    right: np.ndarray,
    order: int,
    *,
    fwhm: float,
    snr_thresh: float,
    ncoeff: int,
    trim_edg: float,
    box_factor: float,
    max_objects: int | None,
) -> tuple[list[ObjectTrace], np.ndarray, np.ndarray, np.ndarray]:
    """Detect and trace objects within one order, returning QA diagnostics."""
    nspec = image.shape[0]
    width = np.nanmax(right - left)
    if not np.isfinite(width) or width < 3:
        return [], np.array([]), np.array([]), np.array([], dtype=int)
    nsamp = int(min(512, max(5, np.ceil(width))))
    rect, spat = rectify_order(image, left, right, nsamp)
    valid = np.isfinite(rect)
    if mask is not None:
        mask_rect, _ = rectify_order(np.asarray(mask, dtype=float), left, right, nsamp)
        valid &= mask_rect > 0.5
    rect_ivar, _ = rectify_order(ivar, left, right, nsamp)
    rect_ivar = np.nan_to_num(rect_ivar, nan=0.0, posinf=0.0, neginf=0.0)

    profile, good = collapse_profile(rect, gpm=valid)
    snr = profile_snr(rect_ivar, good, profile)
    peaks, peak_snr = find_profile_peaks(
        profile,
        snr=snr,
        threshold=snr_thresh,
        fwhm=fwhm,
        trim_edg=trim_edg,
        max_objects=max_objects,
    )

    traces: list[ObjectTrace] = []
    for k, peak in enumerate(peaks):
        positions, xmid, xscale = trace_object(
            rect, spat, valid, int(peak), half_width=fwhm, maxshift=1.0
        )
        coeff, rms = fit_trace(positions, ncoeff=ncoeff)
        failed = not np.all(np.isfinite(coeff))
        centre = float(
            evaluate_trace(coeff, xmid, xscale, np.array([nspec // 2]))[0]
        ) if not failed else float(spat[nspec // 2, int(peak)])
        traces.append(
            ObjectTrace(
                order=order,
                coeff=coeff,
                xmid=xmid,
                xscale=xscale,
                rms=rms,
                fracpos=float((peak + 0.5) / nsamp),
                spat_pixpos=centre,
                fwhm=float(fwhm),
                box_radius=float(box_factor * fwhm),
                snr=float(peak_snr[k]),
                masked=failed,
            )
        )
    return traces, profile, snr, peaks


def find_objects(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Detect spatial object profiles and trace them per order (step 6).

    Each order is rectified along its edge traces, collapsed into a
    sigma-clipped spatial profile, and searched for peaks above the S/N
    threshold. The detected centroids are traced along dispersion and fit with
    a Legendre polynomial. Background pixels outside the object boxes are
    flagged so :func:`subtract_sky` can fit the sky around the objects.
    """
    image = products.get("science")
    if image is None:
        raise ValueError(
            "find_objects requires a 'science' product; run process_science first"
        )
    traces_product = products.get("order_traces")
    if traces_product is None:
        raise ValueError(
            "find_objects requires an 'order_traces' product; run trace_orders first"
        )
    image = np.asarray(image, dtype=float)
    nspec, nspat = image.shape
    ivar = products.get("science_ivar")
    if ivar is None:
        ivar = np.where(np.isfinite(image), 1.0, 0.0)
    ivar = np.asarray(ivar, dtype=float)
    science_mask = products.get("science_mask")
    science_mask = (
        None if science_mask is None else np.asarray(science_mask, dtype=bool)
    )

    fwhm = float(params.get("fwhm", 3.0))
    snr_thresh = float(params.get("snr_thresh", 5.0))
    ncoeff = int(params.get("obj_ncoeff", 4))
    trim_edg = float(params.get("obj_trim_edg", 1.0))
    box_factor = float(params.get("box_factor", 4.0))
    max_objects = params.get("max_objects_per_order")
    max_objects = int(max_objects) if max_objects is not None else None

    existing_table = products.get("objects")
    target_index = _target_order_index(params)
    per_order: dict[int, list[ObjectTrace]] = {}
    if target_index is not None and existing_table is not None:
        per_order = _objects_from_table(existing_table)
        indices = [target_index]
    else:
        indices = list(range(FIRE_NORDERS))

    diagnostics: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    masked_orders: dict[int, str] = {}
    for index in indices:
        order = int(FIRE_ORDER_NUMBERS[index])
        edges = _order_edges(traces_product, order, nspec)
        if edges is None:
            per_order[order] = []
            masked_orders[order] = "no order trace"
            continue
        left, right = edges
        detected, profile, snr, peaks = _detect_order_objects(
            image,
            ivar,
            science_mask,
            left,
            right,
            order,
            fwhm=fwhm,
            snr_thresh=snr_thresh,
            ncoeff=ncoeff,
            trim_edg=trim_edg,
            box_factor=box_factor,
            max_objects=max_objects,
        )
        per_order[order] = detected
        diagnostics[order] = (profile, snr, peaks)

    ordered_traces: list[ObjectTrace] = []
    for order in FIRE_ORDER_NUMBERS:
        ordered_traces.extend(per_order.get(int(order), []))

    rows = np.arange(nspec)
    object_traces = np.full((nspec, len(ordered_traces)), np.nan, dtype=float)
    object_skymask = np.ones((nspec, nspat), dtype=bool)
    for j, trace in enumerate(ordered_traces):
        if trace.masked:
            continue
        positions = evaluate_trace(trace.coeff, trace.xmid, trace.xscale, rows)
        object_traces[:, j] = positions
        radius = max(1.0, trace.box_radius)
        for row in range(nspec):
            centre = positions[row]
            if not np.isfinite(centre):
                continue
            lo = max(0, int(np.floor(centre - radius)))
            hi = min(nspat, int(np.ceil(centre + radius)) + 1)
            if hi > lo:
                object_skymask[row, lo:hi] = False

    table = _objects_table(ordered_traces, ncoeff)
    object_orders = np.asarray([t.order for t in ordered_traces], dtype=np.int16)
    object_box_radii = np.asarray([t.box_radius for t in ordered_traces], dtype=np.float32)

    profiles_figure = _object_profile_figure(diagnostics)
    traces_figure = _object_traces_figure(image, object_traces, object_orders)

    n_masked = int(np.asarray(table["masked"], dtype=bool).sum())
    return StepResult(
        products={
            "objects": table,
            "object_traces": object_traces,
            "object_orders": object_orders,
            "object_box_radii": object_box_radii,
            "object_skymask": object_skymask,
        },
        meta={
            "n_objects": int(len(ordered_traces)),
            "n_objects_masked": n_masked,
            "object_fwhm": fwhm,
            "object_snr_thresh": snr_thresh,
        },
        plots={"object_profiles": profiles_figure, "object_traces": traces_figure},
        masked_orders=masked_orders,
    )


def _object_profile_figure(
    diagnostics: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> go.Figure:
    figure = go.Figure()
    if not diagnostics:
        figure.update_layout(title="Object spatial profiles")
        return figure
    index = max(diagnostics, key=lambda o: len(diagnostics[o][2]))
    profile, snr, peaks = diagnostics[index]
    samples = np.arange(profile.size)
    figure.add_trace(
        go.Scatter(x=samples, y=profile, mode="lines", name="profile")
    )
    figure.add_trace(
        go.Scatter(x=samples, y=snr, mode="lines", name="S/N")
    )
    if peaks.size:
        figure.add_trace(
            go.Scatter(
                x=peaks,
                y=profile[peaks],
                mode="markers",
                marker={"color": "red", "size": 6},
                name="peaks",
            )
        )
    figure.update_layout(
        title=f"Object spatial profile (order {index})",
        xaxis_title="spatial sample",
        yaxis_title="flux",
    )
    return figure


def _object_traces_figure(
    image: np.ndarray, object_traces: np.ndarray, object_orders: np.ndarray
) -> go.Figure:
    nspec = image.shape[0]
    stride = max(1, nspec // 200)
    figure = go.Figure(
        data=go.Heatmap(
            z=np.asarray(image, dtype=float),
            colorscale="Gray",
            colorbar={"title": "counts"},
        )
    )
    rows = np.arange(0, nspec, stride)
    for j in range(object_traces.shape[1]):
        positions = object_traces[rows, j]
        valid = np.isfinite(positions)
        figure.add_trace(
            go.Scatter(
                x=positions[valid],
                y=rows[valid],
                mode="markers",
                marker={"size": 2},
                name=f"order {int(object_orders[j])}",
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Object traces",
        xaxis_title="spatial",
        yaxis_title="dispersion",
    )
    return figure


def _sky_qa_figure(
    panels: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
) -> go.Figure:
    from plotly.subplots import make_subplots

    if not panels:
        figure = go.Figure()
        figure.update_layout(title="Sky model QA")
        return figure
    titles: list[str] = []
    for order, *_ in panels:
        titles.extend([f"order {order} data", f"order {order} model", f"order {order} residual"])
    figure = make_subplots(
        rows=len(panels),
        cols=3,
        subplot_titles=titles,
    )
    for row, (_, data, model, residual) in enumerate(panels, start=1):
        figure.add_trace(
            go.Heatmap(z=data, colorscale="Viridis", showscale=False), row=row, col=1
        )
        figure.add_trace(
            go.Heatmap(z=model, colorscale="Viridis", showscale=False), row=row, col=2
        )
        figure.add_trace(
            go.Heatmap(z=residual, colorscale="RdBu", showscale=False), row=row, col=3
        )
    figure.update_layout(title="Sky model QA", height=350 * len(panels))
    return figure


def subtract_sky(
    products: dict[str, Any],
    meta: dict[str, Any],
    params: dict[str, Any],
) -> StepResult:
    """Model and subtract the 2D sky background per order (step 7).

    The sky is fit as a B-spline along the tilted spectral coordinate times a
    polynomial along the fractional spatial coordinate, using only background
    pixels (outside the object boxes). A local residual surface is then fit
    around each traced object. The resulting sky model is subtracted and the
    propagated variance is emitted alongside the sky-subtracted image.
    """
    image = products.get("science")
    if image is None:
        raise ValueError(
            "subtract_sky requires a 'science' product; run process_science first"
        )
    traces_product = products.get("order_traces")
    if traces_product is None:
        raise ValueError(
            "subtract_sky requires an 'order_traces' product; run trace_orders first"
        )
    image = np.asarray(image, dtype=float)
    nspec, nspat = image.shape
    ivar = products.get("science_ivar")
    ivar = (
        np.asarray(ivar, dtype=float)
        if ivar is not None
        else np.where(np.isfinite(image), 1.0, 0.0)
    )
    science_var = products.get("science_var")
    science_var = (
        np.asarray(science_var, dtype=float) if science_var is not None else 0.0 * image
    )
    science_mask = products.get("science_mask")
    skymask = products.get("object_skymask")
    if skymask is not None:
        skymask = np.asarray(skymask, dtype=bool)

    tilt_coeffs = products.get("tilt_coeffs")
    spec_order = int(meta.get("tilt_spec_order", 3))
    spat_order = int(meta.get("tilt_spat_order", 4))
    spat_scale = float(meta.get("tilt_spat_scale", 10.0))

    bspacing = float(params.get("sky_bspacing", max(15.0, nspec / 12.0)))
    degree = int(params.get("sky_degree", 3))
    npoly = int(params.get("sky_npoly", 2))
    trim_edg = float(params.get("sky_trim_edg", 3.0))
    do_local = bool(params.get("local_sky", True))
    local_factor = float(params.get("local_window_factor", 1.5))

    object_traces = products.get("object_traces")
    object_orders = products.get("object_orders")
    object_box_radii = products.get("object_box_radii")

    sky_global = np.zeros_like(image)
    sky_fit_var = np.zeros_like(image)
    sky_pixels = np.zeros_like(image, dtype=bool)
    n_sky_orders = 0

    qa_panels: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    cols = np.arange(nspat, dtype=float)[None, :]

    for index, order in enumerate(FIRE_ORDER_NUMBERS):
        order = int(order)
        edges = _order_edges(traces_product, order, nspec)
        if edges is None:
            continue
        left, right = edges
        fit = None
        if tilt_coeffs is not None and index < len(tilt_coeffs):
            coeffs = np.asarray(tilt_coeffs[index], dtype=float)
            if np.all(np.isfinite(coeffs)):
                fit = build_tilt_fit(
                    TiltFit(
                        coeffs=coeffs,
                        spec_order=spec_order,
                        spat_order=spat_order,
                        spec_scale=float(max(1, nspec - 1)),
                        spat_scale=spat_scale,
                        rms=float("nan"),
                        nlines=0,
                        tilt_angle=float("nan"),
                        masked=False,
                    )
                )
        spec_coord = tilted_coordinates(
            nspec, nspat, left, right, fit, spat_scale=spat_scale
        )
        width = right - left
        safe_width = np.where(width > 0, width, 1.0)
        on = (cols >= left[:, None] + trim_edg) & (cols <= right[:, None] - trim_edg)
        ximg = (cols - left[:, None]) / safe_width[:, None]
        good = on & np.isfinite(image) & (ivar > 0)
        if skymask is not None:
            good &= skymask
        if good.sum() < max(30, degree + npoly + 4):
            continue

        weights = np.sqrt(np.where(ivar > 0, ivar, 0.0))
        surface = fit_sky_surface(
            spec_coord[good],
            ximg[good],
            image[good],
            weights=weights[good],
            bspacing=bspacing,
            degree=degree,
            npoly=npoly,
        )
        sky_global[on] = surface.evaluate(spec_coord[on], ximg[on])
        if np.isfinite(surface.rms):
            # Added variance from the scatter of the sky-fit residuals.
            sky_fit_var[on] = surface.rms**2
        n_sky_orders += 1

        if (
            do_local
            and object_traces is not None
            and object_orders is not None
            and object_traces.shape[1] > 0
        ):
            members = np.where(np.asarray(object_orders, dtype=int) == order)[0]
            for j in members:
                positions = np.asarray(object_traces[:, j], dtype=float)
                if not np.any(np.isfinite(positions)):
                    continue
                radius = (
                    float(object_box_radii[j]) * local_factor
                    if object_box_radii is not None
                    else 8.0
                )
                near = np.abs(cols - positions[:, None]) <= radius
                window = near & on & np.isfinite(image) & (ivar > 0)
                if skymask is not None:
                    window &= skymask
                if window.sum() < 20:
                    continue
                residual = image - sky_global
                local_surface = fit_sky_surface(
                    spec_coord[window],
                    ximg[window],
                    residual[window],
                    weights=weights[window],
                    bspacing=max(bspacing / 2.0, 5.0),
                    degree=degree,
                    npoly=min(1, npoly),
                )
                target = near & on
                sky_global[target] += local_surface.evaluate(
                    spec_coord[target], ximg[target]
                )

        sky_pixels |= on & (skymask if skymask is not None else True)

        masked_image = np.where(on, image, np.nan)
        masked_model = np.where(on, sky_global, np.nan)
        masked_residual = np.where(on, image - sky_global, np.nan)
        qa_panels.append((order, masked_image, masked_model, masked_residual))

    science_sky = image - sky_global
    science_sky_var = science_var + sky_fit_var
    good_final = np.isfinite(science_sky)
    if science_mask is not None:
        good_final &= np.asarray(science_mask, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        sky_ivar = np.where(
            good_final,
            1.0 / np.maximum(science_sky_var, 1e-12),
            0.0,
        )

    qa = _sky_qa_figure(qa_panels)
    residual_scatter = (
        float(np.sqrt(np.mean(science_sky[sky_pixels] ** 2)))
        if sky_pixels.any()
        else float("nan")
    )
    return StepResult(
        products={
            "sky_model": sky_global,
            "sky_fit_var": sky_fit_var,
            "science_sky": science_sky,
            "science_sky_var": science_sky_var,
            "science_sky_ivar": sky_ivar,
        },
        meta={
            "sky_bspacing": bspacing,
            "sky_npoly": npoly,
            "sky_local": bool(do_local),
            "n_orders_sky_subtracted": int(n_sky_orders),
            "sky_residual_rms": residual_scatter,
        },
        plots={"sky_model": qa},
    )


def register_engines() -> None:
    """Register the implemented Step Engines (reduction steps 1-7)."""
    register_step_engine("prep_calibrations", prep_calibrations)
    register_step_engine("trace_orders", trace_orders)
    register_step_engine("calibrate_wavelengths", calibrate_wavelengths)
    register_step_engine("fit_tilts", fit_tilts)
    register_step_engine("process_science", process_science)
    register_step_engine("find_objects", find_objects)
    register_step_engine("subtract_sky", subtract_sky)
