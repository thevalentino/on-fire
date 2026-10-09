kind: step
title: telluric_correct
purpose: Remove the absorption imprinted by Earth's atmosphere from the flux-calibrated spectra, using either an observed A0V standard star or a precomputed atmospheric model.
order: 100

## Inputs and outputs

**Reads**

- `wave_soln`, `objects`, and the calibrated `flux` and `flux_var` from `flux_calibrate`.
- For the Empirical Telluric Correction: `standard_flux` (the observed standard) and `standard_true` (its true spectrum on each order's grid, written by `flux_calibrate`).
- For the Grid Telluric Correction: the PCA atmospheric components, either as a `telluric_pca` product or from the file named in `telpca_path`.
- Frame Table role loaded for this step: `standard`.

Both corrections are computed whenever their inputs exist. One of them is then made the active result.

**Writes**

| Product | Contents |
|---------|----------|
| `telluric_flux`, `telluric_var`, `telluric_ivar`, `telluric_mask` | The corrected spectra, with one column per object, from the **active** method. |
| `telluric_transmission`, `telluric_wave` | The transmission divided out, and the wavelength grid, from the active method. |
| `telluric` | Table with one row per object: `order`, `obj_id`, `method`, `rms`, `ncoeff` and `masked`. |
| `telluric_empirical_*`, `telluric_grid_*` | The same set of products for each method that could be computed, so both can be compared. |

Metadata recorded: `telluric_method`, `n_objects_telluric_corrected`, `telluric_available_empirical` and `telluric_available_grid`.

**QA Artifacts:** `telluric_transmission` (empirical and grid curves per object) and `telluric_spectra` (corrected spectra over the uncorrected ones).

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until telluric_correct
```

**Parameter Config**

```yaml
steps:
  telluric_correct:
    method: auto                  # auto, empirical or grid (default auto)
    telluric_continuum_order: 6   # empirical: polynomial order of the smooth response (default 6)
    telluric_object_order: 4      # grid: polynomial order of the object continuum (default 4)
    telluric_grid_niter: 4        # grid: continuum/transmission iterations (default 4)
    # telluric_ncoeff: 8          # grid: number of PCA components to fit (default all)
    # telpca_path: telpca.fits    # grid: file holding the PCA components
```

With `method: auto` the Empirical Telluric Correction is active when a standard star is available, and otherwise the Grid Telluric Correction. Asking for a method whose inputs are missing raises `ValueError`.

**Python**, comparing the two and choosing one:

```python
fig = baseline.compare_telluric()          # both corrections, side by side
clone = baseline.run_step("telluric_correct", params={"method": "grid"}, clone=True)
baseline.commit(clone)                     # adopt the grid correction
```

Choosing a method is an ordinary re-run on a State Clone followed by Commit. This step always processes all objects; it takes no `order` argument.

## Figures

{{figure: telluric_correct/transmission "Empirical and grid transmission curves from the telluric_transmission QA Artifact"}}

{{figure: telluric_correct/corrected "Corrected spectrum over the uncorrected one"}}

## Numerical methods and idea

Both methods estimate an atmospheric transmission $T(\lambda)$ and divide the spectrum by it. Pixels are never divided by less than $T = 10^{-3}$.

**Empirical (from an A0V standard).** An A0V star has a nearly featureless spectrum apart from its hydrogen lines, which the template supplies. The ratio of the observed standard to that template is the smooth instrument response times the atmosphere. A robust smooth log-polynomial of order `telluric_continuum_order` is fit to the ratio and divided out, leaving

$$
T(\lambda) = \frac{\text{observed standard} / \text{A0V template}}{\text{smooth response}},
$$

clipped to the range $[10^{-3}, 1]$. Absolute scaling, exposure time and sampling cancel in the ratio.

**Grid (from a PCA model).** The atmosphere is represented by a TelPCA grid whose components linearly describe the inverse hyperbolic sine of the optical depth:

$$
\operatorname{arcsinh}\tau(\lambda) = \sum_k a_k\, C_k(\lambda), \qquad
T(\lambda) = e^{-\tau(\lambda)},
$$

where $C_0$ is the model mean and the coefficients $a_k$ scale the other components. With no standard star, the object's own continuum and the transmission are fit together by alternating `telluric_grid_niter` times. Each pass fits a robust polynomial continuum of order `telluric_object_order` to the spectrum divided by the current transmission (after the first pass, using only pixels where the transmission is above 0.9, so absorption lines do not bias it), then refits the PCA coefficients by linear least squares to the spectrum divided by that continuum.

**Applying.** The corrected flux is $F/T$ and the variance becomes $\sigma^2/T^2$.

## Where this happens in the code

The Step Engine picks the active method and assembles the products:

{{code: onfire/core/engines.py::telluric_correct}}

The pure array algorithms in `onfire/core/telluric.py`:

{{code: onfire/core/telluric.py::empirical_transmission}}

{{code: onfire/core/telluric.py::pca_transmission}}

{{code: onfire/core/telluric.py::fit_telluric_grid}}

{{code: onfire/core/telluric.py::apply_telluric}}
