title: Quickstart
order: 2

# Quickstart

The whole reduction is two commands plus one file you edit by hand. Paths below are
placeholders; the walkthrough with real data and output comes later.

## 1. Build the Frame Table

Point `onfire setup` at a directory of raw `*.fits` exposures:

```bash
onfire setup /path/to/raw -o frames.ascii
```

This reads each header, assigns every exposure a role, and writes the
[Frame Table](frame-table.html). It refuses to overwrite an existing table unless
you pass `--overwrite`. Open the file and correct any roles it guessed wrong.

## 2. Write a Parameter Config

The config can be nearly empty, because every key has a default. A minimal file:

```yaml
frames: frames.ascii
output_dir: checkpoints
qa_dir: qa
```

See [Parameter Config](parameter-config.html) for per-step overrides.

## 3. Reduce

```bash
onfire reduce config.yaml
```

onfire runs the steps in order and prints one line per completed step. When it
finishes it reports the last step reached.

{{figure: quickstart/reduce-output "Terminal output of a full onfire reduce run"}}

### Stopping and resuming

```bash
# stop after the wavelength solution, then inspect the results
onfire reduce config.yaml --until calibrate_wavelengths

# continue from the next step, using the saved state
onfire reduce config.yaml --from fit_tilts
```

`--until` and `--from` accept a step name or its 1-based number. `--from` needs the
checkpoint file of the step *before* the one you name, so an earlier `--until`
run makes it available.

### Other options

| Option | Effect |
|--------|--------|
| `--frames PATH` | Use this Frame Table instead of the config's `frames` value. |
| `-o`, `--output-dir DIR` | Write checkpoint files here instead of the config's `output_dir`. |
| `--fail-fast` | Halt on the first failure and print code to reproduce it. |
| `--interactive` | Open an IPython session bound to the state when the run stops. |
| `--no-qa` | Skip writing the Plotly QA HTML files. |

## What you get

- Checkpoint files in `output_dir`, named `NN_stepname.fits` (for example
  `02_trace_orders.fits`), one per completed step.
- QA Artifacts in `qa_dir`, as standalone HTML you can open in a browser.

Next: how the pipeline is organised in each step, starting with
[Trace orders](trace_orders.html).
