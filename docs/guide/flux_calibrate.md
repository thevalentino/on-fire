kind: step
title: flux_calibrate
purpose: Use an observed standard star of known spectrum to derive a sensitivity function for each order, and apply it to turn extracted counts into flux density.
order: 90

## Inputs and outputs

**Reads**

- `wave_soln` from `calibrate_wavelengths`.
- `objects` from `find_objects`, and the extracted spectra from `extract_1d`: `opt_flux` and `opt_var` if present, otherwise `boxcar_flux` and `boxcar_var`.
- `standard_flux`: the **extracted standard-star spectrum**, shape `(nspec, 21)` with one column per order, in counts. Optionally `standard_var` with the same shape.
- Optionally `standard_template`, a table with `wave` and `flux` columns giving the standard's true spectrum. Without it the bundled A0V (Vega) template is used, or the file in `a0v_path`.
- Frame Table role loaded for this step: `standard`.

> **Note:** the `standard_flux` product is not produced by any earlier step. The runner loads the standard-star exposures as `raw_standard`, but nothing in the pipeline currently extracts them into `standard_flux`. Until that exists, put your own extracted standard spectrum into the state before running this step, for example `state.products["standard_flux"] = counts`. The tests do exactly this.

**Writes**

| Product | Contents |
|---------|----------|
| `sensfunc` | Table with one row per order: `npoints`, `rms`, `wave_min`, `wave_max`, `coeff` and `masked`. |
| `sensitivity` | The conversion factor at every pixel, shape `(nspec, 21)`. |
| `standard_fluxed` | The standard star after calibration, for checking against `standard_true`. |
| `standard_true` | The true spectrum of the standard on each order's wavelength grid. |
| `flux`, `flux_var`, `flux_ivar`, `flux_mask` | The calibrated science spectra, with one column per object like the `extract_1d` matrices. |

An order is added to the Order Mask with the reason `no standard star spectrum` or `sensitivity fit failed`. Objects in a masked order get no calibrated flux. The input `standard_flux` is also passed through unchanged as an output product.

**QA Artifacts:** `sensitivity` and `flux_calibration`.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until flux_calibrate
```

**Parameter Config**

```yaml
steps:
  flux_calibrate:
    standard_exptime: 60.0   # exposure time of the standard, seconds (default 1.0)
    science_exptime: 900.0   # exposure time of the science frames, seconds (default 1.0)
    sensfunc_order: 5        # degree of the sensitivity polynomial per order (default 5)
    sensfunc_niter: 8        # rejection iterations (default 8)
    sensfunc_sigrej: 3.0     # rejection threshold in robust sigma (default 3.0)
    # a0v_path: my_vega.fits # replace the bundled A0V template
```

The exposure times are **not** read from the Frame Table or the FITS headers. If you leave them at the default of 1.0 second, the calibrated fluxes are off by the ratio of the real times.

`standard_airmass` (default 1.0) is accepted, but the engine does not give `derive_sensitivity` an extinction curve, so it currently has no effect on the result.

**Python**, on a State Clone:

```python
clone = baseline.run_step("flux_calibrate", params={"sensfunc_order": 4}, clone=True)
fig = baseline.compare(clone, "flux_calibrate")
```

This step always processes all orders. The engine does not read `order=`, so a targeted call would re-run every order anyway.

## Figures

{{figure: flux_calibrate/sensitivity "Sensitivity function of each order from the sensitivity QA Artifact"}}

{{figure: flux_calibrate/calibrated-standard "Calibrated standard star over its true spectrum"}}

## Numerical methods and idea

A standard star has a known spectrum. Dividing what the instrument recorded by what the star really emits gives the instrument's response, and that response then converts any other spectrum from counts to flux.

**1. Counts to a rate per Angstrom.** On each order's wavelength grid,

$$
N_\lambda = \frac{\text{counts}}{t_\mathrm{exp}\,\Delta\lambda},
$$

where $\Delta\lambda$ is the absolute gradient of the wavelength solution at each pixel.

**2. Zeropoint.** The true flux $F_\lambda$ of the standard (the template interpolated onto the order's wavelengths) gives a per-pixel AB-style zeropoint,

$$
Z = -2.5 \log_{10}\!\left(\frac{\lambda^2 F_\lambda}{N_\lambda}\right) + Z_0,
$$

with $Z_0$ a unit-conversion constant fixed by the flux scale of $10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$. Pixels with non-positive counts or ratios, non-finite values or no template coverage are left out.

**3. Fit the sensitivity function.** The zeropoints of each order are fit with a Legendre series of degree `sensfunc_order` against wavelength scaled to $[-1, 1]$, with iterative `sensfunc_sigrej`-sigma rejection for up to `sensfunc_niter` iterations.

**4. Apply.** For a science spectrum, the same rate $N_\lambda$ is formed with the science exposure time and multiplied by the conversion factor,

$$
F_\lambda = N_\lambda \,\frac{10^{-0.4\,(Z(\lambda) - Z_0)}}{\lambda^2},
$$

and the variance is scaled by the square of the whole linear factor, $\left(\text{factor} / (t_\mathrm{exp}\,\Delta\lambda)\right)^2$. A pixel is valid only where the factor is positive. The factor is zero where the fitted zeropoint falls outside 5 to 30.

**To verify:** the output flux unit. The engine docstring says erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$, but the zeropoint constant is built with a $10^{-17}$ scale, so the numbers may carry that scale factor.

## Where this happens in the code

The Step Engine derives one sensitivity function per order and applies it to every object:

{{code: onfire/core/engines.py::flux_calibrate}}

Choosing the standard's true spectrum:

{{code: onfire/core/engines.py::_standard_template}}

The pure array algorithms in `onfire/core/flux.py`:

{{code: onfire/core/flux.py::counts_to_nlam}}

{{code: onfire/core/flux.py::compute_zeropoint}}

{{code: onfire/core/flux.py::fit_sensitivity_function}}

{{code: onfire/core/flux.py::apply_sensitivity}}
