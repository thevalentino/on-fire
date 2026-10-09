kind: step
title: trace_orders
purpose: Locate and follow the left and right spatial edges of each of the 21 echelle orders (11 to 31) on the flat, and fit each edge with a smooth curve.
order: 20

## Inputs and outputs

**Reads**

- `flat_norm` (or `flat` if no normalised flat exists): the combined flat from `prep_calibrations`, shape `(nspec, nspat)`.
- Frame Table role loaded for this step: `pixelflat`.

**Writes**

| Product | Contents |
|---------|----------|
| `order_traces` | Table with one row per order: `order`, `left_cen`, `right_cen`, `left_coeff`, `right_coeff`, `xmid`, `xscale`, `rms_left`, `rms_right`, `masked`. |
| `edge_traces` | Image of shape `(nspec, 2 * 21)` with each edge's fitted spatial position per row. |

**QA Artifact:** `edge_traces`, the edges overlaid on the flat.

An order whose edge fit fails is flagged in the Order Mask with the reason
`edge trace fit failed`; the other orders are unaffected.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until trace_orders

# later, continue with the next step from the saved state
onfire reduce config.yaml --from calibrate_wavelengths
```

**Parameter Config**

```yaml
steps:
  trace_orders:
    ncoeff: 5       # number of polynomial coefficients per edge (default 5)
    maxshift: 3.0   # largest row-to-row edge movement, in pixels (default 3.0)
```

**Python**, one order only, on a State Clone:

```python
clone = baseline.run_step("trace_orders", params={"ncoeff": 6}, clone=True, order=13)
fig = baseline.compare(clone, "trace_orders")
```

With `order=` set and an existing `order_traces` product, only that order is
re-traced and every other order keeps its previous fit. An order outside 11 to 31
raises `ValueError`.

## Figures

{{figure: trace_orders/edge-fit "Fitted order edges overlaid on the normalised flat"}}

{{figure: trace_orders/edge-fit-zoom "Zoom on one order showing left and right edge fits"}}

## Numerical methods and idea

The goal is two smooth curves per order, one for the left edge and one for the
right, that later steps use to know where each order sits on the detector.

**1. Find edges at the reference row.** The flat is collapsed along dispersion
(median over rows) and smoothed with a 7-pixel boxcar. The peak of this spatial profile
near each expected order position is taken as the order centre. The left edge is
the strongest *rising* spatial gradient between the previous order and the centre.
The right edge is the strongest *falling* gradient between the centre and the next
order. Each search covers at most half the gap to the neighbouring order (the smaller
gap, for an order with neighbours on both sides).

**2. Follow each edge along dispersion.** Starting from the reference row (the
middle of the detector) and working outward in both directions, each edge position
is recentred row by row. The new position is the centroid of the positive part of
the spatial gradient within `maxshift` pixels of the position in the adjacent row.
For the left edge the gradient is used as is, and for the right edge its sign is
flipped, so both edges look like a rising edge to the centroid.

**3. Fit a smooth curve.** With the row coordinate rescaled to $x \in [-1, 1]$,

$$
x = \frac{\text{row} - x_\mathrm{mid}}{x_\mathrm{scale}}, \qquad
x_\mathrm{mid} = \frac{n_\mathrm{spec} - 1}{2},
\qquad
x_\mathrm{scale} = x_\mathrm{mid} \ (\text{or } 1 \text{ if } x_\mathrm{mid} = 0),
$$

each tracked edge $y(x)$ is fit with a Legendre series of `ncoeff` terms,

$$
y(x) \approx \sum_{k=0}^{\mathrm{ncoeff}-1} c_k\, P_k(x),
$$

by ordinary least squares over the rows where the trace is finite. The reported
`rms_left` and `rms_right` are the root-mean-square residuals of the fit. An edge
with fewer finite rows than coefficients is not fit and masks its order.

**To verify:** the expected order positions are the fixed fractional positions
stored for FIRE (`FIRE_ORDER_SPAT_POS`); this page has not checked how those values
were derived.

## Where this happens in the code

The Step Engine in the pipeline layer gathers parameters, handles the single-order
case, and builds the output table:

{{code: onfire/core/engines.py::trace_orders}}

The pure array algorithms it calls live in `onfire/core/tracing.py`, independent of
any state handling:

{{code: onfire/core/tracing.py::detect_order_edges}}

{{code: onfire/core/tracing.py::trace_columns}}

{{code: onfire/core/tracing.py::fit_edge_traces}}
