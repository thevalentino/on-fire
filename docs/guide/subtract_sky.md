kind: step
title: subtract_sky
purpose: Model the two-dimensional sky background of each order, following the line tilt and avoiding the objects, and subtract it from the science image.
order: 70

## Inputs and outputs

**Reads**

- `science`, `science_ivar`, `science_var` and `science_mask` from `process_science`.
- `order_traces` from `trace_orders`.
- `tilt_coeffs` from `fit_tilts`. The tilt surface sizes are read from the metadata that step recorded.
- `object_skymask`, `object_traces`, `object_orders` and `object_box_radii` from `find_objects`.
- No Frame Table role is loaded for this step.

**Writes**

| Product | Contents |
|---------|----------|
| `sky_model` | The fitted sky, same shape as the science image. |
| `science_sky` | The science image minus the sky model. |
| `science_sky_var` | Variance of `science_sky`: the science variance plus the scatter of the sky fit. |
| `science_sky_ivar` | Inverse variance, `0` where the pixel is not usable. |
| `sky_fit_var` | The extra variance contributed by the sky fit. |

Metadata recorded: `sky_bspacing`, `sky_npoly`, `sky_local`, `n_orders_sky_subtracted` and `sky_residual_rms`.

**QA Artifact:** `sky_model`, with the image, the sky model and the residual for each order.

An order with no usable trace, or too few sky pixels to fit, is skipped without being added to the Order Mask. Check `n_orders_sky_subtracted` in the metadata.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until subtract_sky
```

**Parameter Config**

```yaml
steps:
  subtract_sky:
    sky_bspacing: 30.0          # spacing of the B-spline breakpoints, in pixels (default max(15, nspec/12))
    sky_degree: 3               # B-spline degree, 3 is cubic (default 3)
    sky_npoly: 2                # Legendre degree across the order (default 2)
    sky_trim_edg: 3.0           # pixels to ignore inside each order edge (default 3.0)
    local_sky: true             # add a local correction around each object (default true)
    local_window_factor: 1.5    # width of that local window, in object box radii (default 1.5)
```

**Python**

```python
clone = baseline.run_step("subtract_sky", params={"sky_bspacing": 20.0}, clone=True)
fig = baseline.compare(clone, "subtract_sky")
```

This step always processes all orders; it takes no `order` argument.

## Figures

{{figure: subtract_sky/sky-model "Image, sky model and residual of one order from the sky_model QA Artifact"}}

## Numerical methods and idea

Telluric OH lines dominate the near-infrared sky, and they are tilted on the detector. A good model has to follow them along the line, not along the detector columns.

**1. Tilted coordinate.** For every pixel of an order, the dispersion row is corrected by the tilt model from `fit_tilts`. The result is the row the pixel would have if it sat on the order centre line:

$$
s = \text{row} - \delta(\text{row},\ \text{dspat}),
$$

so a sky line has the same $s$ at every position across the slit. Without a tilt surface the plain row is used.

**2. Fractional spatial coordinate.** Each pixel also gets $u = (\text{col} - \text{left}) / (\text{right} - \text{left})$, between 0 at the left edge and 1 at the right.

**3. Choose sky pixels.** Pixels are used if they lie at least `sky_trim_edg` inside the order edges, are finite with positive inverse variance, and fall outside every object box (`object_skymask`).

**4. Fit the surface.** The sky is modelled as a product of a cubic B-spline in $s$, with breakpoints every `sky_bspacing` pixels, and a Legendre polynomial in $u$:

$$
\text{sky}(s, u) = \sum_{a,b} c_{ab}\, B_a(s)\, P_b(u).
$$

The coefficients come from weighted least squares with weights $\sqrt{\text{ivar}}$ and iterative 3-sigma rejection, up to eight iterations. The surface is evaluated for the whole order, including under the objects. An order with fewer than `max(30, degree + npoly + 4)` sky pixels is skipped.

**5. Local correction.** If `local_sky` is on, a second, smaller fit is made to the residual in a window of `local_window_factor` times the object's box radius around each traced object, using finer breakpoints (half of `sky_bspacing`, but at least 5 pixels) and at most a linear term across the order. It is added to the sky model there.

**6. Subtract and propagate.** The sky model is subtracted from the science image. The residual variance of the sky fit, the squared RMS of its residuals, is added to the science variance.

## Where this happens in the code

The Step Engine loops over orders and combines the global and local fits:

{{code: onfire/core/engines.py::subtract_sky}}

The pure array algorithms in `onfire/core/skysub.py`:

{{code: onfire/core/skysub.py::tilted_coordinates}}

{{code: onfire/core/skysub.py::fit_sky_surface}}
