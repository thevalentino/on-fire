kind: step
title: process_science
purpose: Flat-field the science exposures, propagate their noise, and either difference nodded pairs (A minus B) or average stare frames.
order: 50

## Inputs and outputs

**Reads**

- `raw_science`: the stack of science exposures, shape `(nframes, nspec, nspat)`.
- `flat_norm` (or `flat`) from `prep_calibrations`.
- `bad_pixel_mask` from `prep_calibrations`, if present.
- Frame Table role loaded for this step: `science`.

**Writes**

| Product | Contents |
|---------|----------|
| `science` | The flat-fielded, combined science image. |
| `science_var` | Its variance, propagated from photon and read noise. |
| `science_ivar` | Inverse variance, `0` for bad pixels. |
| `science_mask` | Boolean good-pixel mask. |

Metadata recorded: `science_mode`, `n_science_frames`, `n_science_combined` and `n_science_bad_pixels`.

**QA Artifact:** `science`, a heatmap of the combined image.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until process_science
```

**Parameter Config.** The observing mode is a top-level key, because it also affects later steps:

```yaml
mode: pair_difference     # or: stare (default pair_difference)
steps:
  process_science:
    gain: 1.0             # detector gain, electrons per count (default 1.0)
    read_noise: 0.0       # detector read noise, electrons (default 0.0)
```

**Python**

```python
state = state.run_step("process_science", params={"mode": "stare"})
```

This step works on the whole detector; it takes no `order` argument. An unknown `mode` raises `ValueError`.

## Figures

{{figure: process_science/science-frame "Combined science image from the science QA Artifact"}}

## Numerical methods and idea

The two observing modes differ in how the sky background is dealt with, and that decides what this step has to do.

**Flat-fielding.** Each exposure is divided by the normalised flat. Pixels where the flat is non-finite or not positive are set to zero instead of producing infinities.

**Noise.** The variance of a flat-fielded frame pushes the photon and read noise through that division:

$$
\sigma^2_\mathrm{frame} = \frac{\max(C, 0)/g + R^2}{F^2},
$$

where $C$ is the raw counts, $g$ is `gain`, $R$ is `read_noise` and $F$ is the flat response.

**Pair Differencing** (`mode: pair_difference`). Consecutive exposures are taken as nodded pairs, frames 1 and 2, 3 and 4, and so on. Each pair is flat-fielded and subtracted, $A - B$, which cancels the thermal and OH sky emission to first order. The pair variance is $\sigma^2_A + \sigma^2_B$. The differences are averaged, and with $N$ pairs the variance of that mean is

$$
\sigma^2_\mathrm{science} = \frac{1}{N^2} \sum_{p=1}^{N} \left(\sigma^2_{A,p} + \sigma^2_{B,p}\right).
$$

At least two frames are required. An odd final frame is not used.

**Stare Mode** (`mode: stare`). Each frame is flat-fielded and the frames are averaged, with the variance of the mean computed the same way. No sky is removed here, so the 2D sky model in `subtract_sky` has to do that work.

**Masking.** A pixel is good if the science value and variance are finite, the variance is positive and the pixel is not in `bad_pixel_mask`. The inverse variance is $1/\max(\sigma^2, 10^{-12})$ for good pixels and 0 otherwise.

## Where this happens in the code

The Step Engine chooses the mode and builds the products:

{{code: onfire/core/engines.py::process_science}}

The pure array functions are in `onfire/core/science.py`:

{{code: onfire/core/science.py::flat_field}}

{{code: onfire/core/science.py::frame_variance}}

{{code: onfire/core/science.py::pair_difference}}
