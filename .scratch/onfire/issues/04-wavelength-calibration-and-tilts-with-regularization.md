# 04: Wavelength Calibration and Tilts with Regularization

**What to build:**
Implementation of wavelength calibration (Step 3) and spectral line tilt modeling (Step 4). Bundles canonical Magellan/FIRE reference files (reidentification template and OH airglow line lists), identifies arc/OH emission lines, fits 1D dispersion solutions per order, and models 2D spectral line tilts. In automated runs, it fits a global surface across orders; in interactive mode, it supports re-running a single order (e.g. order 13) with cross-order regularization from the global solution, outputting interactive Plotly residual and line identification diagnostics.

**Blocked by:** 03: Calibrations Prep and Order Edge Tracing

**Status:** ready-for-agent

- [x] Bundles and references FIRE wavelength archive templates and line lists without external network requirements.
- [x] Performs line detection and 1D wavelength calibration across all 21 orders (`calibrate_wavelengths`).
- [x] Models 2D spectral tilts along orders and fits spatial-spectral distortion surfaces (`fit_tilts`).
- [x] Supports interactive single-order execution with cross-order regularization, as well as full global recompute.
- [x] Generates interactive Plotly QA plots showing wavelength fit residuals and arc line identifications.
- [x] High-level tests verify line matching, tilt angle derivation, and single-order re-execution.
