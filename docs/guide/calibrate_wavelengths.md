kind: step
title: calibrate_wavelengths
purpose: Find the emission lines in the arc spectrum of every order, identify them against the bundled FIRE archive, and fit a wavelength solution per order plus one global solution across orders.
order: 30

## Inputs and outputs

**Reads**

- `order_traces` from `trace_orders`, to know where each order lies.
- `raw_arcs` (or an existing `arc_image`): the arc exposures, averaged into one image.
- Frame Table role loaded for this step: `arc`.
- The bundled FIRE archive spectra and OH/arc line list, unless you point to your own files (see `template_path` and `line_list_path` below).

**Writes**

| Product | Contents |
|---------|----------|
| `arc_image` | The combined arc image. |
| `arc_spectra`, `order_centers`, `order_halfwidths` | The extracted 1D arc spectrum per order, and where in the image it was taken. |
| `wavelengths` | Table with one row per order, including its fit quality and `masked` flag. |
| `wave_soln` | Wavelength at every dispersion pixel for every order, shape `(nspec, 21)`. |
| `wave_coeffs` | Legendre coefficients of each order's 1D solution. |
| `wave_surface`, `wave_surface_coeffs` | The global 2D pixel-versus-order solution and its coefficients. |
| `line_ids` | Table of every identified line: `order`, `pixel`, `wave`. |

An order is added to the Order Mask with the reason `no arc spectrum` (no usable trace or no archive template for it) or `insufficient line identifications`.

**QA Artifacts:** `wavelength_residuals` and `wavelength_identifications`.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until calibrate_wavelengths
```

**Parameter Config**

```yaml
steps:
  calibrate_wavelengths:
    ncoeff: 5            # coefficients of each order's stored 1D solution (default 5)
    arc_box_rad: 3.0     # half-width in pixels of the box used to extract the arc spectrum (default 3.0)
    sigdetect: 5.0       # line-detection threshold in noise sigma (default 5.0)
    fwhm: 4.0            # expected line FWHM in pixels (default 4.0)
    match_toler: 2.0     # matching tolerance in pixels of dispersion (default 2.0)
    n_first: 2           # polynomial order of the first-pass fit (default 2)
    n_final: 4           # polynomial order of the final fit (default 4)
    nspec_coeff: 4       # global surface: coefficients along the pixel axis (default 4)
    norder_coeff: 6      # global surface: coefficients along the order axis (default 6)
    # template_path: my_archive.fits   # optional replacement for the bundled archive
    # line_list_path: my_lines.dat     # optional replacement for the bundled line list
```

**Python**, re-doing one order that failed, on a State Clone:

```python
clone = baseline.run_step(
    "calibrate_wavelengths", params={"sigdetect": 4.0}, clone=True, order=13
)
fig = baseline.compare(clone, "calibrate_wavelengths")
```

A targeted run needs a previous wavelength solution to refine. Without one, `order=` has no effect and every order is solved.

## Figures

{{figure: calibrate_wavelengths/residuals "Line-fit residuals per order from the wavelength_residuals QA Artifact"}}

{{figure: calibrate_wavelengths/identifications "Arc spectrum of one order with identified lines marked"}}

## Numerical methods and idea

The idea is to avoid identifying lines from scratch. Each order has an archived arc spectrum with a known wavelength scale. The step aligns the observed spectrum to that archive, uses the alignment to guess each line's wavelength, and snaps the guess to the nearest true line in the line list.

**1. Extract an arc spectrum per order.** The arc image is summed over a box of $\pm$`arc_box_rad` pixels around the order centre, which is the midpoint of the fitted left and right edges from `trace_orders`, row by row.

**2. Detect lines.** The continuum is estimated with a running median filter, iteratively clipped so lines do not pull it up. After subtracting it, the noise is the robust scatter $1.4826 \times \mathrm{MAD}$. Local maxima above `sigdetect` times that noise, separated by at least one `fwhm`, are kept as lines. Each line centre is the flux-weighted centroid within $\pm 1.25\,$`fwhm` pixels.

**3. Align to the archive.** The observed and archived spectra are cross-correlated (using Fourier transforms) to find a pixel shift. The shift maps each detected line to an archive pixel, and the archive's wavelength scale turns that into a guessed wavelength.

**4. Match to the line list.** Each guess is matched to the nearest rest wavelength in the line list, accepted if closer than `match_toler` pixels' worth of wavelength. The tolerance is `match_toler` times the archive dispersion.

**5. Fit and refine.** The matched lines are fit with a Legendre polynomial of order `n_first` in pixel, with 3-sigma robust rejection. The fit is then used to predict where all detected lines should fall, new matches are added, and the fit is repeated with a higher order, up to `n_final`, for at most six iterations. An order with fewer than three detected lines, or fewer than three matches, is masked.

**6. Global surface.** All identified lines from all orders are fit together with a two-dimensional Legendre surface in pixel and order number, with `nspec_coeff` terms along pixel and `norder_coeff` along order (coordinates scaled to $[0,1]$). This needs at least `nspec_coeff * norder_coeff` matched lines; otherwise the stored surface is just a copy of the per-order solutions.

**Single-order re-runs and Cross-Order Regularization.** When `order` is given and a solution exists, the order is not solved from the archive again. Its detected lines are matched against the existing global surface as the prior (or the previous per-order solution if there is no surface). A smooth low-order correction to the prior is fit to the line residuals, so a faint order inherits the instrument-wide solution instead of failing. The matching tolerance is the larger of `match_toler` and 3.0. With fewer than three matches the prior is used unchanged. The global surface is not refit in a single-order run, so `wave_surface` stays as it was, and `line_ids` then lists only the target order's lines.

## Where this happens in the code

The Step Engine handles the full-run and single-order branches and assembles the products:

{{code: onfire/core/engines.py::calibrate_wavelengths}}

The pure algorithms in `onfire/core/wavelengths.py`:

{{code: onfire/core/wavelengths.py::detect_lines}}

{{code: onfire/core/wavelengths.py::xcorr_shift}}

{{code: onfire/core/wavelengths.py::reidentify_spectrum}}

{{code: onfire/core/wavelengths.py::regularize_wavelength_solution}}

{{code: onfire/core/wavelengths.py::fit_wavelength_surface}}
