kind: step
title: fit_tilts
purpose: Measure how arc lines lean across the slit in every order and fit a smooth two-dimensional model of that tilt.
order: 40

## Inputs and outputs

**Reads**

- `arc_image`, `arc_spectra`, `order_centers` and `order_halfwidths`, all produced by `calibrate_wavelengths`.
- Frame Table role loaded for this step: `arc`.

**Writes**

| Product | Contents |
|---------|----------|
| `tilts` | Table with one row per order: `nlines`, `rms`, `tilt_angle` and `masked`. |
| `tilt_coeffs` | Legendre coefficients of each order's tilt surface. |
| `tilt_surface_coeffs` | Coefficients of one global surface pooled across all orders. |
| `tilt_measurements` | Every measurement: `order`, `spec_pixel`, `dspat`, `delta` and the fitted `model`. |

An order is masked with the reason `no arc spectrum`, `too few arc lines for tilts` (fewer than three detected lines) or `tilt tracing failed` (fewer than four usable measurements).

**QA Artifacts:** `tilt_residuals` (fit residuals against dispersion pixel) and `tilt_map` (the tilt model for the first valid order).

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until fit_tilts
```

**Parameter Config**

```yaml
steps:
  fit_tilts:
    spec_order: 3         # Legendre terms along the spectral axis (default 3)
    spat_order: 4         # Legendre terms along the spatial axis (default 4)
    tilt_sigdetect: 10.0  # line-detection threshold in noise sigma (default 10.0)
    fwhm: 4.0             # expected line FWHM in pixels (default 4.0)
    tilt_maxshift: 3.0    # largest line movement between neighbouring columns, in pixels (default 3.0)
    # spat_scale: 10.0    # spatial normalisation in pixels; default is the median order half-width minus 2
```

**Python**, one order, on a State Clone:

```python
clone = baseline.run_step("fit_tilts", params={"tilt_sigdetect": 7.0}, clone=True, order=13)
fig = baseline.compare(clone, "fit_tilts")
```

A targeted run needs earlier tilt coefficients to refine. Without them `order=` has no effect and every order is fit.

## Figures

{{figure: fit_tilts/tilt-map "Tilt model of one order from the tilt_map QA Artifact"}}

{{figure: fit_tilts/tilt-residuals "Tilt-fit residuals from the tilt_residuals QA Artifact"}}

## Numerical methods and idea

In a curved echelle image a line of constant wavelength is not exactly along the spatial direction. It leans, and the lean varies with position. Sky subtraction and extraction later need to follow that lean, so this step measures it from arc lines and models it.

**1. Pick lines.** In each order the arc spectrum is searched for lines above `tilt_sigdetect` noise sigma, using the same line detector as `calibrate_wavelengths`.

**2. Trace each line across the slit.** Starting at the order centre, each line's spectral position is followed one spatial column at a time in both directions. At each column the new position is the centroid of the positive flux within `tilt_maxshift` pixels of the previous column's position. A line is dropped if it has fewer than three samples, or drifts by more than $3\,\mathrm{fwhm} + 1$ pixels.

Each sample is recorded as a pair: the spatial offset `dspat` from the order centre, and the spectral shift $\delta$ from the line's position at the centre.

**3. Fit a surface.** With the spectral pixel scaled by $n_\mathrm{spec} - 1$ and the spatial offset by `spat_scale`, the shift is fit with a two-dimensional Legendre surface:

$$
\frac{\delta}{n_\mathrm{spec}-1} \approx \sum_{j,k} c_{jk}\, P_j(s)\, P_k(u),
\qquad
s = \frac{\text{spec pixel}}{n_\mathrm{spec}-1}, \quad u = \frac{\text{dspat}}{\text{spat\_scale}},
$$

with `spec_order` terms in $j$ and `spat_order` terms in $k$, and 3-sigma robust rejection. The quoted tilt angle is the arctangent of the local slope $\partial\delta/\partial(\text{dspat})$ at the middle of the order.

**4. Global surface.** The measurements from all orders are also pooled into one global surface, stored as `tilt_surface_coeffs`.

**Single-order re-runs (Cross-Order Regularization).** With `order=` the target order is not fit freely. A low-order correction (one term per axis) to the global surface is fit to that order's residuals instead. A faint order then inherits the instrument-wide distortion instead of being dropped, and an order with fewer than four measurements uses the global surface unchanged.

## Where this happens in the code

The Step Engine:

{{code: onfire/core/engines.py::fit_tilts}}

The pure algorithms in `onfire/core/tilts.py`:

{{code: onfire/core/tilts.py::trace_line_tilt}}

{{code: onfire/core/tilts.py::fit_tilt_surface}}

{{code: onfire/core/tilts.py::regularize_tilt_fit}}
