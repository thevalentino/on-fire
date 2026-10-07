# 03: Calibrations Prep and Order Edge Tracing

**What to build:**
Implementation of the first two canonical reduction steps: calibration frame preparation and order edge tracing. This extracts FIRE optical edge-tracing logic from PypeIt into pure functional engines, combining raw flats, building detector bad pixel masks, locating order traces, and tracing physical spatial boundaries for all 21 echelle orders (orders 11–31). It saves Multi-Extension FITS checkpoints and interactive Plotly QA edge overlays, and supports inspecting or re-tracing individual orders interactively.

**Blocked by:** 02: ReductionState Container, Multi-Extension FITS I/O, and State Cloning

**Status:** ready-for-agent

- [ ] Combines raw flat exposures and constructs bad pixel masks (`prep_calibrations`).
- [ ] Traces left and right spatial edges across detector dispersion for orders 11 through 31 (`trace_orders`).
- [ ] Supports re-running edge tracing on an individual order (e.g. order 13) in interactive mode.
- [ ] Emits intermediate FITS checkpoints for calibration products and order edge trace sets.
- [ ] Generates interactive Plotly QA figures displaying detector images with overlaid order traces.
- [ ] High-level tests verify flat combination, order detection, and trace model outputs.
