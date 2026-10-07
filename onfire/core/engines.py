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
from onfire.core.tracing import (
    detect_order_edges,
    evaluate_edge_trace,
    fit_edge_traces,
    trace_coordinate,
    trace_columns,
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


def register_engines() -> None:
    """Register the calibration prep and order tracing Step Engines."""
    register_step_engine("prep_calibrations", prep_calibrations)
    register_step_engine("trace_orders", trace_orders)
