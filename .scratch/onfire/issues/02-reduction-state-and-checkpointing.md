# 02: ReductionState Container, Multi-Extension FITS I/O, and State Cloning

**What to build:**
The core state management container (`ReductionState`) that serves as the central data model and interactive execution boundary. It orchestrates step execution, handles state persistence via Multi-Extension FITS checkpoints with chronological file prefixes, provides in-memory State Cloning (`clone=True`) to allow experimental parameter exploration without mutating baseline states, and attaches interactive Plotly figures to an inspectable dictionary.

**Blocked by:** 01: Package Scaffolding, Frame Ingestion, and Setup Validation

**Status:** ready-for-agent

- [x] `ReductionState` initializes from an input Frame Table and YAML Parameter Config.
- [x] Supports executing steps in-place or returning an isolated State Clone (`clone=True`).
- [x] Persists intermediate data arrays, masks, and metadata to Multi-Extension FITS checkpoints with step prefixes (e.g. `01_...fits`).
- [x] Reloads previously saved checkpoints into `ReductionState` without information loss.
- [x] Attaches Plotly `Figure` objects to `state.plots` and supports saving standalone HTML QA files.
- [x] High-level tests verify state mutation, cloning independence, and FITS save/load roundtripping.
