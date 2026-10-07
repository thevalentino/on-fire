# 05: Science Processing and Sky Subtraction

**What to build:**
Implementation of science frame reduction (Step 5), object finding (Step 6), and 2D background sky modeling (Step 7). Supports both Pair Differencing (A - B nodding) for thermal background and OH subtraction, as well as Stare Mode for single non-dithered exposures. Detects spatial object profiles within each order, traces spatial centroids, and performs 2D B-spline global and local sky subtraction, generating interactive Plotly QA heatmaps of sky models and residuals.

**Blocked by:** 04: Wavelength Calibration and Tilts with Regularization

**Status:** ready-for-agent

- [ ] Applies flat-fielding, bad pixel masking, and pair differencing (A - B) or single-frame stare prep (`process_science`).
- [ ] Detects spatial profiles, establishes object traces, and defines background extraction slits per order (`find_objects`).
- [ ] Models and subtracts 2D sky background using 2D B-splines with tilted line coordinates (`subtract_sky`).
- [ ] Emits intermediate FITS checkpoints with sky-subtracted 2D images and variance arrays.
- [ ] Generates interactive Plotly QA figures showing 2D data, model, and residual heatmaps per order.
- [ ] High-level tests verify both Pair Differencing and Stare Mode workflows, object detection, and residual statistics.
