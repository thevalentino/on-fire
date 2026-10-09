kind: step
title: extract_1d
purpose: Turn the sky-subtracted image into one-dimensional spectra, one per object and order, with both a simple boxcar sum and an optimal (profile-weighted) extraction.
order: 80

## Inputs and outputs

**Reads**

- `science_sky` and `science_sky_ivar` from `subtract_sky`, and `science_mask` from `process_science`.
- `objects` and `object_traces` from `find_objects`.
- `order_traces` from `trace_orders`.
- No Frame Table role is loaded for this step.

**Writes**

The spectra are matrices with one column per object in `objects`, in the same order, and `nspec` rows.

| Product | Contents |
|---------|----------|
| `extractions` | Table with one row per object: the profile fit and per-object summaries. |
| `boxcar_flux`, `boxcar_var`, `boxcar_ivar`, `boxcar_npix`, `boxcar_mask` | The boxcar spectrum, its variance and inverse variance, pixels summed, and validity. |
| `opt_flux`, `opt_var`, `opt_ivar`, `opt_chi2`, `opt_frac_use`, `opt_mask` | The optimal spectrum, its variance and inverse variance, a per-row reduced chi-squared, the profile fraction used, and validity. |
| `extraction_cr_mask` | Image of pixels rejected as cosmic rays. |

Metadata recorded: `n_objects_extracted`, `n_orders_extracted`, `n_cosmic_rays` and the profile and rejection settings used.

An object that is masked in `objects`, or has no finite trace, gets an empty column. An object in an order with no usable trace also adds `no order trace` to the Order Mask.

**QA Artifacts:** `extraction_spectra`, `extraction_noise` and `extraction_profiles`.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until extract_1d
```

**Parameter Config**

```yaml
steps:
  extract_1d:
    profile_model: gaussian    # spatial profile shape: gaussian or polynomial (default gaussian)
    profile_order: 2           # Legendre terms when profile_model is polynomial (default 2)
    min_frac_use: 0.5          # least profile fraction a row must retain to be valid (default 0.5)
    cr_sigclip: 5.0            # cosmic-ray threshold in sigma; null turns rejection off (default 5.0)
    cr_niter: 3                # cosmic-ray rejection passes (default 3)
    # box_radius: 6.0          # boxcar half-width in pixels; default is each object's own box radius
    # profile_half_width: 6.0  # half-width of the profile window; default max(3, box radius)
```

**Python**, one order, on a State Clone:

```python
clone = baseline.run_step("extract_1d", params={"profile_model": "polynomial"}, clone=True, order=13)
fig = baseline.compare(clone, "extract_1d")
```

With `order=` and an existing `extractions` table that has one row per object, only that order is re-extracted and the others keep their results.

## Figures

{{figure: extract_1d/spectra "Boxcar and optimal spectra of one object from the extraction_spectra QA Artifact"}}

{{figure: extract_1d/profile "Spatial profile and its fit from the extraction_profiles QA Artifact"}}

## Numerical methods and idea

Two extractions are made for every object because they trade simplicity against signal-to-noise.

**Boxcar.** For each dispersion row the usable pixels within `box_radius` of the object trace are summed, and their variances are added. No assumption about the object's shape is needed, but pixels with no flux still add noise.

**Optimal extraction (Horne 1986).** The object's spatial profile is measured once and then used to weight each row.

1. *Collapse the profile.* Every row contributes the pixels within `profile_half_width` of the trace, weighted by inverse variance, and these are averaged by integer offset from the trace.
2. *Fit it.* A Gaussian is fit by a least-squares parabola to the logarithm of the profile (using samples above 5% of the peak), or, with `profile_model: polynomial`, a Legendre series of `profile_order` terms.
3. *Normalise per row.* The fitted profile is placed on the trace and divided by its sum over the unmasked pixels, so each row of $P$ sums to one.
4. *Weight.* With data $D$, inverse variance $w$ and profile $P$ in a row,

$$
f = \frac{\sum_x w\,P\,D}{\sum_x w\,P^2}, \qquad
w_f = \frac{\sum_x w\,P^2}{\sum_x P},
$$

where $f$ is the extracted flux and $w_f$ is its inverse variance.

**Cosmic rays.** After each pass the model $f\,P$ is subtracted. A pixel is rejected if its positive residual exceeds `cr_sigclip` times the larger of its formal noise and the robust scatter of all residuals. The extraction is repeated, up to `cr_niter` times, until nothing new is rejected.

**Validity.** A row of the optimal spectrum is kept only if more than `min_frac_use` of the profile survived the masking.

## Where this happens in the code

The Step Engine loops over objects and handles the single-order case:

{{code: onfire/core/engines.py::extract_1d}}

One object, both methods:

{{code: onfire/core/engines.py::_extract_object}}

The pure array algorithms in `onfire/core/extract.py`:

{{code: onfire/core/extract.py::boxcar_extract}}

{{code: onfire/core/extract.py::collapse_object_profile}}

{{code: onfire/core/extract.py::optimal_extract}}
