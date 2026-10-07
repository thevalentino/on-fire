# 01: Package Scaffolding, Frame Ingestion, and Setup Validation

**What to build:**
A functioning package scaffold and an automated setup CLI/API tool that inspects a directory of raw Magellan/FIRE FITS files, reads detector and observation header metadata, automatically classifies frame types (science, standard, pixelflat, arc), validates that required calibrations are present, and generates a human-readable, fixed-width ASCII Frame Table.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Standard Python package structure is created with executable console script entrypoints.
- [ ] Running the setup command against a directory of raw Magellan/FIRE files parses headers (`OBSTYPE`, `OBJECT`, `EXPTIME`, `ACQTIME`, `AIRMASS`, etc.) and categorizes frames.
- [ ] Setup validates input files and raises clear warnings if required calibration frames (flats, arcs) are missing.
- [ ] Generates a clean, fixed-width ASCII Frame Table (`frames.ascii`) that can be viewed and edited in standard text editors.
- [ ] High-level tests verify header ingestion, frame classification, and fixed-width table output formatting.
