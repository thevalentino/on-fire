title: Parameter Config
order: 4

# The Parameter Config

The Parameter Config is a YAML file. Whatever you do not set takes a default, so
the file only needs the things you want to change. It is kept separate from the
Frame Table on purpose: the Frame Table says *which data*, the config says *how to
reduce it*.

## Pipeline defaults

| Key | Default | Meaning |
|-----|---------|---------|
| `mode` | `pair_difference` | How science frames are background-subtracted: `pair_difference` or `stare`. |
| `frames` | `frames.ascii` | Frame Table path. |
| `output_dir` | `checkpoints` | Where checkpoint files are written. |
| `qa_dir` | `qa` | Where QA Artifact HTML files are written. |
| `steps` | `{}` | Per-step parameter overrides (below). |

## Step overrides

Parameters for one step live under `steps:`, keyed by the step name:

```yaml
mode: pair_difference
steps:
  trace_orders:
    ncoeff: 6
    maxshift: 2.5
```

Which parameters each step accepts is listed on that step's page, under *How to
call it*.

## How a call is assembled

When a step runs, its parameters are layered, later layers winning:

1. the top-level config keys (except `steps`, `qa_dir`, `frames`, `output_dir`),
2. the step's own block under `steps:`,
3. any parameters you pass explicitly in Python,
4. the `order` you target, if any.

You can see what a step would be called with, without running it:

```python
state.resolve_call_params("trace_orders")
```

{{code: onfire/pipeline/state.py::ReductionState.resolve_call_params}}

## Loading from Python

{{code: onfire/pipeline/config.py::load_config}}
