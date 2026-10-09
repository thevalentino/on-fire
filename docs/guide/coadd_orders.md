kind: step
title: coadd_orders
purpose: Splice the calibrated spectra of the individual orders into one continuous spectrum, combining the overlaps by inverse-variance weighting.
order: 110

## Inputs and outputs

**Reads**

- `wave_soln` from `calibrate_wavelengths` and `objects` from `find_objects`.
- The spectra to combine: `telluric_flux`, `telluric_var` and `telluric_mask` from `telluric_correct` if present, otherwise `flux`, `flux_var` and `flux_mask` from `flux_calibrate`.
- No Frame Table role is loaded for this step.

**Writes**

| Product | Contents |
|---------|----------|
| `coadd_wave` | Wavelength grid of the combined spectrum, in Angstroms. |
| `coadd_flux`, `coadd_var`, `coadd_ivar` | Flux density, variance and inverse variance. |
| `coadd_mask` | Boolean mask of valid samples. |

Metadata recorded: `n_coadd_orders`, `n_coadd_points` and `coadd_obj_id`.

**QA Artifact:** `coadd_orders`, the combined spectrum with its uncertainty.

This step combines every order into one spectrum, so the product has no order axis to trim. If any order's input is Stale, the whole coadd is Stale, and Commit discards it as a whole instead of order by order.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until coadd_orders
```

**Parameter Config**

```yaml
steps:
  coadd_orders:
    # obj_id: 3     # combine only this object (default: the highest-S/N object of each order)
```

**Python**

```python
clone = baseline.run_step("coadd_orders", clone=True)
fig = baseline.compare(clone, "coadd_orders")
```

This step takes no `order` argument.

**Choosing the object.** By default one object is taken from each order, the one with the highest S/N in the `objects` table. Giving `obj_id` restricts the candidates to that one object. Because `find_objects` numbers objects uniquely across all orders, a given `obj_id` belongs to a single order, so the coadd then contains only that order. **To verify:** whether that is the intended use of `obj_id`.

## Figures

{{figure: coadd_orders/coadd "Combined spectrum and its uncertainty from the coadd_orders QA Artifact"}}

## Numerical methods and idea

Neighbouring echelle orders overlap slightly in wavelength, so the same part of the spectrum is measured twice with different noise. The step merges all orders onto one wavelength grid and weights each measurement by how well it is known.

1. For each order, samples with a non-finite wavelength or flux, a masked sample, or a non-positive variance are dropped. Each remaining sample gets the weight $w = 1/\sigma^2$.
2. The output grid is regular, from the shortest to the longest wavelength present, with a step equal to the **finest** median pixel spacing among the orders.
3. Every sample is assigned to the bin containing its wavelength. No resampling or interpolation is done. The samples are used as measured.
4. In each bin the combined flux, variance and inverse variance are

$$
\bar f = \frac{\sum_i w_i f_i}{\sum_i w_i}, \qquad
\sigma^2 = \frac{1}{\sum_i w_i}, \qquad
\text{ivar} = \sum_i w_i .
$$

Bins that received no samples are left out of the output, so `coadd_wave` is not necessarily a gapless grid.

If a variance array is missing, every sample is weighted equally.

## Where this happens in the code

The Step Engine chooses the spectra and the object of each order:

{{code: onfire/core/engines.py::coadd_orders}}

{{code: onfire/core/engines.py::_select_order_object}}

The pure array algorithm in `onfire/core/coadd.py`:

{{code: onfire/core/coadd.py::coadd_order_spectra}}
