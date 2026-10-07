# 06: 1D Spectral Extraction

**What to build:**
Implementation of 1D spectral extraction (Step 8). Performs both standard boxcar extraction and Horne optimal extraction for each detected object trace in orders 11 through 31, fully propagating formal variance arrays and bad pixel flags. Allows interactive parameter tweaking per order (e.g. extraction aperture width, profile model order), persisting intermediate FITS tables and interactive Plotly 1D spectrum and spatial profile QA plots.

**Blocked by:** 05: Science Processing and Sky Subtraction

**Status:** ready-for-agent

- [x] Extracts 1D flux, variance, and inverse variance using standard boxcar extraction (`extract_1d`).
- [x] Extracts 1D spectra using optimal extraction with profile fitting and cosmic ray rejection per order (`extract_1d`).
- [x] Supports interactive re-extraction and parameter adjustment for an individual order in IPython.
- [x] Persists extracted 1D multi-order spectra into structured Multi-Extension FITS tables.
- [x] Generates interactive Plotly QA plots displaying 1D spectra, noise vectors, and spatial profile fits.
- [x] High-level tests verify flux conservation, variance propagation, and optimal extraction weighting.
