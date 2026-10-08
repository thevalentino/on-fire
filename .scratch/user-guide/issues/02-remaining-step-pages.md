# 02: Write the remaining 11 step pages

**What to build:** One page per remaining Reduction Step in `STEP_SEQUENCE`, following the template approved in ticket 01: purpose, inputs/outputs, how to call it (CLI, Parameter Config, Python), figure placeholders, collapsed numerical-methods section drafted from the engine in `onfire/core/`, and collapsed code-snippet section using by-reference extraction. Steps: `prep_calibrations`, `calibrate_wavelengths`, `fit_tilts`, `process_science`, `find_objects`, `subtract_sky`, `extract_1d`, `flux_calibrate`, `telluric_correct` (cover Empirical and Grid Telluric Correction), `coadd_orders`, `extinction_correct`. Cross-order steps should say why they cannot be trimmed order by order.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Every step in `STEP_SEQUENCE` has a page; sidebar is in pipeline order
- [ ] Each page follows the fixed template with both sections collapsed by default
- [ ] Methods text matches what the code does; gaps are marked "to verify" and listed in the ticket's `## Comments` for author review
- [ ] Parameter Config keys shown on each page exist in the real config schema
- [ ] `tests/test_docs.py` passes, including committed output equals fresh build
