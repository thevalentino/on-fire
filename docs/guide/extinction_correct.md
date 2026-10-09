kind: step
title: extinction_correct
purpose: Remove the reddening caused by Milky Way dust along the line of sight, giving the final intrinsic spectrum.
order: 120

## Inputs and outputs

**Reads**

- `coadd_wave`, `coadd_flux` and `coadd_var` from `coadd_orders`.
- The colour excess $E(B-V)$: either the `ebv` parameter, or looked up from a `dust_map` product at the target coordinates.
- The target coordinates, if looking up: `ra` and `dec` parameters, or `ra` and `dec` in the state's metadata. A run through `onfire reduce` fills the metadata from the science FITS header, in degrees, trying the card pairs `RA`/`DEC`, `RA_DEG`/`DEC_DEG`, `TELRA`/`TELDEC` and `OBJRA`/`OBJDEC` in that order.
- No Frame Table role is loaded for this step.

**Writes**

| Product | Contents |
|---------|----------|
| `final_wave` | Wavelength grid, unchanged from `coadd_wave`. |
| `final_flux` | The dereddened flux density. |
| `final_var`, `final_ivar` | Variance and inverse variance. |
| `final_mask` | Boolean mask of valid samples. |

Metadata recorded: `ebv`, `rv` and `n_extinction_points`.

**QA Artifact:** `extinction_correct`, the dereddened spectrum over the observed one.

This is a cross-order step, like `coadd_orders`: if its input is Stale, the whole product is.

## How to call it

**Command line**

```bash
onfire reduce config.yaml --until extinction_correct
```

**Parameter Config**

```yaml
steps:
  extinction_correct:
    ebv: 0.05      # E(B-V) in magnitudes; skips the dust-map lookup
    rv: 3.1        # total-to-selective extinction ratio (default 3.1)
    # ra: 150.1    # target right ascension in degrees, used only for the lookup
    # dec: 2.2     # target declination in degrees, used only for the lookup
```

With neither `ebv` nor coordinates, the step raises `ValueError`.

> **Note:** nothing in the pipeline currently supplies a `dust_map`. Without one, the lookup raises `ValueError`, so for a command-line reduction set `ebv` yourself in the Parameter Config. To use a lookup from Python, put a table with `ra`, `dec` and `ebv` columns into `state.products["dust_map"]`.

**Python**

```python
clone = baseline.run_step("extinction_correct", params={"ebv": 0.03, "rv": 3.1}, clone=True)
fig = baseline.compare(clone, "extinction_correct")
```

This step takes no `order` argument.

## Figures

{{figure: extinction_correct/dereddened "Dereddened spectrum over the observed one from the extinction_correct QA Artifact"}}

## Numerical methods and idea

Dust between us and the target absorbs and scatters light more strongly at shorter wavelengths. The step reverses that with a standard extinction law and the amount of dust along the line of sight.

**1. Amount of dust.** $E(B-V)$ is taken from the `ebv` parameter if it is given. Otherwise it is read from the dust map as the value of the **nearest** catalogue entry, using a plain squared difference in right ascension and declination. This is not an angular distance, so it is only reliable away from the poles and for a map that is dense near the target.

**2. Extinction law.** The Fitzpatrick (1999) law gives the extinction relative to the V band,

$$
\frac{A(\lambda)}{A(V)} = a(x) + \frac{b(x)}{R_V}, \qquad x = \frac{10^4}{\lambda\,[\text{\AA}]},
$$

with $A(V) = R_V\, E(B-V)$. The code implements the infrared, optical and ultraviolet branches of the law; wavelengths longer than about 9100 Å ($x \le 1.1$) use the infrared branch.

**3. Correct.** The flux is multiplied by the factor

$$
10^{\,0.4\,A(\lambda)},
$$

and the variance by the square of that factor. A sample is valid where the flux is finite and the factor is positive.

## Where this happens in the code

The Step Engine finds $E(B-V)$ and calls the correction:

{{code: onfire/core/engines.py::extinction_correct}}

The pure array algorithms in `onfire/core/extinction.py`:

{{code: onfire/core/extinction.py::query_ebv}}

{{code: onfire/core/extinction.py::extinction_factor}}

{{code: onfire/core/extinction.py::deredden_spectrum}}
