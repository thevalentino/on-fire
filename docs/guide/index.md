title: Overview
order: 1

# onfire: the big picture

onfire reduces raw Magellan/FIRE echelle exposures (orders 11 to 31) into a
calibrated, combined one-dimensional spectrum. It is built to be *inspectable*:
every stage can be stopped, looked at, and re-run.

This guide assumes you already know FIRE data and echelle reduction. It explains
how onfire is organised and how to drive it, not what a flat field is.

## What goes in, what comes out

Two small human-readable files describe a reduction:

| Input | What it holds | Page |
|-------|---------------|------|
| **Frame Table** | One row per raw exposure, with its metadata and the role it plays (science, standard, pixelflat, arc). | [Frame Table](frame-table.html) |
| **Parameter Config** | A YAML file of defaults and per-step parameter overrides. | [Parameter Config](parameter-config.html) |

The pipeline writes an **Intermediate State** after every step (a FITS checkpoint
file) and, unless you pass `--no-qa`, interactive Plotly **QA Artifacts** as
standalone HTML files.

{{figure: overview/pipeline-flow "Frame Table and Parameter Config in, twelve Reduction Steps, final spectrum out"}}

## Twelve Reduction Steps

The canonical sequence runs in this order. Each step takes the Intermediate State
left by the previous one, runs one scientific computation, and leaves a new
Intermediate State.

1. `prep_calibrations`
2. `trace_orders`
3. `calibrate_wavelengths`
4. `fit_tilts`
5. `process_science`
6. `find_objects`
7. `subtract_sky`
8. `extract_1d`
9. `flux_calibrate`
10. `telluric_correct`
11. `coadd_orders`
12. `extinction_correct`

Each step has its own page in the sidebar. Steps whose page is not written yet are
shown greyed out.

## Two ways to work

**Batch.** Run everything from the command line with `onfire reduce`, optionally
stopping with `--until` or resuming with `--from`. See the [Quickstart](quickstart.html).

**Interactive.** Load an Intermediate State in Python, clone it into a **State
Clone**, re-run a step (or a single order) with different parameters, compare
against the **Baseline** in one figure, and **Commit** the clone if it is better.
See [Interactive analysis](interactive.html).

## Orders that fail

A failing order does not stop a batch reduction. It is recorded in the **Order
Mask** with a reason, and the remaining orders carry on. Use `--fail-fast` to halt
at the first failure instead; onfire then saves the state and prints the code to
reproduce the failure interactively.
