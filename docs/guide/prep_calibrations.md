kind: step
title: prep_calibrations
purpose: Combine the raw pixel flats into one master flat, make a normalised copy for edge tracing, and flag bad detector pixels.
order: 10

## Inputs and outputs

**Reads**

- `raw_flats`: the stack of raw flat exposures, shape `(nimages, nspec, nspat)`.
- Frame Table role loaded for this step: `pixelflat`.

**Writes**

| Product | Contents |
|---------|----------|
| `flat` | Combined (mean) flat, shape `(nspec, nspat)`. |
| `flat_var` | Propagated Poisson variance of `flat`. |
| `flat_norm` | `flat` divided by its smooth response along dispersion. This is what `trace_orders` uses. |
| `bad_pixel_mask` | Boolean mask, `True` for bad pixels. |

Metadata recorded: `n_flats`, `n_bad_pixels`, `flat_good` (every pixel had at least one usable exposure) and `flat_nused_median` (median number of exposures used per pixel).

**QA Artifact:** `flat`, a heatmap titled "Master flat".

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until prep_calibrations
```

**Parameter Config**

```yaml
steps:
  prep_calibrations:
    sigma_clip: true         # reject outlier pixels between exposures (default true)
    sigrej: 2.0              # rejection threshold in robust sigma (default depends on image count)
    normalize_window: 101    # smoothing width along dispersion, in pixels (default 101)
    bad_pixel_sigma: 5.0     # bad-pixel threshold in robust sigma (default 5.0)
    bad_pixel_window: 5      # local smoothing width for bad pixels, in pixels (default 5)
```

**Python**

```python
state = state.run_step("prep_calibrations", params={"normalize_window": 151})
```

This step always works on the whole detector; it takes no `order` argument.

## Figures

{{figure: prep_calibrations/master-flat "Master flat heatmap from the QA Artifact"}}

{{figure: prep_calibrations/bad-pixels "Bad-pixel mask over a section of the detector"}}

## Numerical methods and idea

This step turns several noisy flat exposures into three things later steps need: a clean flat, an estimate of its noise, and a list of pixels to distrust.

**1. Combine the flats.** With three or more exposures and `sigma_clip` on, each pixel is compared with the median of that pixel across exposures. The scatter is the robust estimate

$$
\sigma = 1.4826 \,\operatorname{median}\left|f_i - \tilde f\right|,
$$

and an exposure's value is rejected where $|f_i - \tilde f| / \sigma \ge$ `sigrej`. Where $\sigma = 0$ nothing is rejected. If `sigrej` is not given it depends on the number of exposures: 1.1, 1.3, 1.6 and 1.9 for 3, 4, 5 and 6 images, and 2.0 otherwise. With fewer than three exposures no clipping is done. Non-finite values are always excluded.

The combined flat is the mean of the surviving values, and its variance treats each value's magnitude as its own Poisson variance:

$$
\bar f = \frac{\sum_i w_i f_i}{\sum_i w_i}, \qquad
\operatorname{Var}(\bar f) = \frac{\sum_i w_i\,|f_i|}{\left(\sum_i w_i\right)^2},
$$

where $w_i$ is 1 for kept values and 0 for rejected ones.

**2. Normalise.** The flat is divided by a boxcar-smoothed copy of itself, smoothed along the dispersion axis only, with `normalize_window` pixels. Removing the slow spectral response while keeping the spatial structure leaves the sharp order edges that `trace_orders` follows.

**3. Flag bad pixels.** Non-finite and non-positive pixels are flagged first. The flat is then smoothed along the spatial axis with a boxcar of `bad_pixel_window` pixels, and any pixel whose residual from that smooth version exceeds `bad_pixel_sigma` times the robust scatter (median absolute deviation based, over the valid pixels) is flagged too.

## Where this happens in the code

The Step Engine collects parameters, calls the three computations and packages the products:

{{code: onfire/core/engines.py::prep_calibrations}}

The computations are pure array functions in `onfire/core/calibrations.py`:

{{code: onfire/core/calibrations.py::combine_flats}}

{{code: onfire/core/calibrations.py::normalize_flat}}

{{code: onfire/core/calibrations.py::build_bad_pixel_mask}}
