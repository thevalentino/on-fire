kind: step
title: find_objects
purpose: Find the spatial position of every object in every order, trace each one along dispersion, and mark the pixels around them so the sky can be fit away from the objects.
order: 60

## Inputs and outputs

**Reads**

- `science`, `science_ivar` and `science_mask` from `process_science`.
- `order_traces` from `trace_orders`.
- No Frame Table role is loaded for this step.

**Writes**

| Product | Contents |
|---------|----------|
| `objects` | Table with one row per detected object: `order`, `obj_id`, `fracpos`, `spat_pixpos`, `fwhm`, `box_radius`, `snr`, `rms`, `xmid`, `xscale`, `ncoeff`, `trace_coeff` and `masked`. |
| `object_traces` | Spatial position of every object at every dispersion row, shape `(nspec, n_objects)`. |
| `object_orders` | The order number of each object. |
| `object_box_radii` | The extraction box radius of each object. |
| `object_skymask` | Boolean image; `False` inside an object's box, `True` where the sky may be fit. |

`obj_id` is unique across all orders, counting in the order the objects are listed. An order without a usable edge trace is masked with the reason `no order trace`. An object whose trace fit fails stays in the table with its own `masked` flag set.

**QA Artifacts:** `object_profiles` (the spatial profile of each order) and `object_traces` (the traces over the science image).

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until find_objects
```

**Parameter Config**

```yaml
steps:
  find_objects:
    fwhm: 3.0                  # expected object FWHM in pixels (default 3.0)
    snr_thresh: 5.0            # detection threshold on the profile S/N (default 5.0)
    obj_ncoeff: 4              # Legendre coefficients of each object's trace (default 4)
    obj_trim_edg: 1.0          # ignore peaks this many samples from an order edge (default 1.0)
    box_factor: 4.0            # extraction box radius, in units of fwhm (default 4.0)
    # max_objects_per_order: 1 # keep only the brightest N objects per order (default no limit)
```

**Python**, one order, on a State Clone:

```python
clone = baseline.run_step("find_objects", params={"snr_thresh": 3.0}, clone=True, order=13)
fig = baseline.compare(clone, "find_objects")
```

With `order=` and an existing `objects` table, only that order is searched again, and the other orders keep their objects.

## Figures

{{figure: find_objects/profile "Collapsed spatial profile of one order with detected peaks"}}

{{figure: find_objects/traces "Object traces overlaid on the science image"}}

## Numerical methods and idea

Orders are curved and slanted on the detector, so objects are searched for in a straightened copy of each order.

**1. Rectify.** Each order is resampled onto a uniform grid that runs from its left edge to its right edge at every row. The number of samples across is the widest order width in pixels, between 5 and 512.

**2. Collapse to a profile.** Along the dispersion axis, samples are iteratively rejected where they deviate from the column median by more than 5 robust sigma (using $1.4826\times\mathrm{MAD}$). The survivors are averaged into a mean spatial profile, and a quadratic baseline is subtracted so object peaks sit on a flat floor.

**3. Signal-to-noise.** The inverse variance image is rectified in the same way and propagated through the average, giving a S/N value for every sample of the profile.

**4. Find peaks.** The profile and its S/N are smoothed with a Gaussian of $\sigma = \mathrm{fwhm}/2.3548$. Local maxima that are positive and above `snr_thresh` in S/N are kept. Peaks within `obj_trim_edg` samples of an edge are discarded, and peaks closer together than $0.75\,\mathrm{fwhm}$ are thinned, keeping the brightest. If `max_objects_per_order` is set, only that many of the brightest remain.

**5. Trace and fit.** From the middle row, outward in both directions, the object's centroid is measured in a window of $\pm\mathrm{fwhm}$ pixels, and each row is allowed to move at most one pixel from the position expected from the rectified peak. The centroids are fit with a Legendre series of `obj_ncoeff` terms with iterative 3-sigma rejection, the same polynomial family used for the order edges.

**6. Sky mask.** For every object, the pixels within `box_factor` $\times$ `fwhm` of its trace are set to `False` in `object_skymask`. `subtract_sky` fits the sky only where the mask is `True`.

## Where this happens in the code

The Step Engine loops over orders, builds the tables and the sky mask:

{{code: onfire/core/engines.py::find_objects}}

Per-order detection:

{{code: onfire/core/engines.py::_detect_order_objects}}

The pure array algorithms in `onfire/core/objects.py`:

{{code: onfire/core/objects.py::collapse_profile}}

{{code: onfire/core/objects.py::find_profile_peaks}}

{{code: onfire/core/objects.py::trace_object}}
