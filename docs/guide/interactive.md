title: Interactive analysis
order: 5

# Interactive analysis

The batch command is for getting a first reduction. When one order or one step
looks wrong, work in Python. The loop is: **clone, re-run, compare, commit**.

## Load a state

Every completed step leaves a checkpoint file. Load one back into a
`ReductionState`, the container that holds the Intermediate State:

```python
from onfire.pipeline.state import ReductionState

baseline = ReductionState.load("checkpoints/02_trace_orders.fits")
print(baseline.step, baseline.step_name)
```

Or add `--interactive` to `onfire reduce` to get an IPython session already bound
to the final state.

## Clone and re-run

A **State Clone** is an isolated copy. Change it freely; the **Baseline** is never
touched.

```python
clone = baseline.run_step(
    "trace_orders",
    params={"ncoeff": 7},
    clone=True,
    order=13,
)
```

- `params` overrides the Parameter Config for this call only.
- `order=13` re-runs just that order. All other orders keep their previous results.
- `clone=True` runs on a new State Clone and returns it. Without it, `run_step`
  changes the state you call it on.

A successful single-order re-run clears that order's **Order Mask**, since the old
failure no longer applies.

{{code: onfire/pipeline/state.py::ReductionState.run_step}}

## Compare

`compare` builds one Plotly figure with this state on the left and the other on the
right, from the QA Artifacts each recorded for that step:

```python
fig = baseline.compare(clone, "trace_orders")
fig.show()
```

If either side has no QA Artifact for that step it raises `ValueError`, saying
which orders are covered. A step can register its own comparison figure instead of
the default pairing. `telluric_correct` does, to show the Empirical and Grid
Telluric Correction side by side.

## Stale products

Re-running an early step makes later products out of date: they were made with the
old calibration. onfire does not delete them. It marks them **Stale** (per order,
when you targeted one) so you can still look at them. They are discarded at Commit.

## Commit

If the clone is better, adopt it:

```python
baseline.commit(clone)
```

Commit replaces the Baseline **wholesale and in place**. Your existing variables
stay valid, the clone is left unchanged, and any product still marked Stale is
dropped. Recorded progress rewinds to the last step with no Stale products, so you
re-run forward from there.

Nothing is written to disk by Commit. Save when you are ready:

```python
baseline.save("checkpoints")
```

### When Commit refuses

A clone made from a different state, or from a Baseline that has changed since the
clone was made, is refused with `ValueError`. Pass `force=True` to commit anyway.

{{code: onfire/pipeline/state.py::ReductionState.commit}}
