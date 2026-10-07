# 07: Flux Calibration and Telluric Correction Selection

**What to build:**
Implementation of per-order spectrophotometric flux calibration (Step 9) and dual-engine telluric absorption correction (Step 10). Supports deriving sensitivity functions from standard stars, applying empirical A0V standard star telluric modeling, and fitting precomputed TelPCA atmospheric transmission grids. In automated runs, falls back to TelPCA when standard stars are unavailable; in interactive sessions, provides `state.compare_telluric()` to render a dual Plotly comparison figure and allow users to select which correction commits to the state.

**Blocked by:** 06: 1D Spectral Extraction

**Status:** ready-for-agent

- [ ] Computes and applies per-order sensitivity functions from observed standard star exposures (`flux_calibrate`).
- [ ] Corrects telluric molecular absorption per order using an empirical A0V standard star model (`telluric_correct`).
- [ ] Corrects telluric absorption per order by fitting precomputed PCA atmospheric grids (`telluric_correct`).
- [ ] Implements automated fallback logic (empirical if standard exists, else grid) for batch runs.
- [ ] Implements `state.compare_telluric()` generating a dual Plotly QA visualization and enabling interactive selection.
- [ ] High-level tests verify sensitivity curve derivation, telluric feature removal, and interactive selection dispatch.
