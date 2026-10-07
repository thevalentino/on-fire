# 08: Order Splicing, Galactic Dust Extinction, and End-to-End Orchestrator

**What to build:**
Implementation of order splicing (Step 11), Galactic dust extinction correction (Step 12), and the end-to-end pipeline orchestrator (`onfire reduce config.yaml`). Coadds overlapping orders 11–31 into a continuous 1D spectrum, corrects intrinsic fluxes using target coordinates, $E(B-V)$ dust maps, and the Fitzpatrick 1999 extinction law. Implements CLI flags (`--until`, `--from`, `--interactive`), order masking for damaged orders, and `--fail-fast` mode that prints exact IPython commands to reload and reproduce failed steps with identical parameters.

**Blocked by:** 07: Flux Calibration and Telluric Correction Selection

**Status:** ready-for-agent

- [x] Slices and coadds overlapping calibrated orders 11 through 31 into a continuous 1D spectrum (`coadd_orders`).
- [x] Corrects for Milky Way dust extinction via target coordinates ($E(B-V)$ query with local config override) and Fitzpatrick 1999 law (`extinction_correct`).
- [x] Provides the master `onfire reduce` CLI command and top-level Python orchestration runner.
- [x] Supports `--until <step>` to pause execution and `--from <step>` to resume from saved intermediate FITS checkpoints.
- [x] Default batch mode masks failed orders with an Order Mask and completes healthy orders.
- [x] `--fail-fast` mode halts on failure and outputs exact copy-pasteable Python code to reload the state and re-run the failed step with identical arguments.
- [x] High-level tests verify order splicing, dereddening mathematics, CLI flag controls, and fail-fast diagnostic reporting.
