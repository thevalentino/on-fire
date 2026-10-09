# 02: Write the remaining 11 step pages

**What to build:** One page per remaining Reduction Step in `STEP_SEQUENCE`, following the template approved in ticket 01: purpose, inputs/outputs, how to call it (CLI, Parameter Config, Python), figure placeholders, collapsed numerical-methods section drafted from the engine in `onfire/core/`, and collapsed code-snippet section using by-reference extraction. Steps: `prep_calibrations`, `calibrate_wavelengths`, `fit_tilts`, `process_science`, `find_objects`, `subtract_sky`, `extract_1d`, `flux_calibrate`, `telluric_correct` (cover Empirical and Grid Telluric Correction), `coadd_orders`, `extinction_correct`. Cross-order steps should say why they cannot be trimmed order by order.

**Blocked by:** 01

**Status:** resolved

- [x] Every step in `STEP_SEQUENCE` has a page; sidebar is in pipeline order
- [x] Each page follows the fixed template with both sections collapsed by default
- [x] Methods text matches what the code does; gaps are marked "to verify" and listed in the ticket's `## Comments` for author review
- [x] Parameter Config keys shown on each page exist in the real config schema
- [x] `tests/test_docs.py` passes, including committed output equals fresh build

## Comments

Items for author review (also marked "to verify" or called out on the pages):

- **Pipeline gap, not a docs problem:** nothing produces `standard_flux` (needed by `flux_calibrate` and the empirical `telluric_correct`) from the loaded `raw_standard` frames, and nothing supplies `dust_map` for `extinction_correct`. A plain `onfire reduce` therefore stops at `flux_calibrate` with a `ValueError`. The tests inject `standard_flux` directly. The quickstart and the two step pages say so. Needs its own ticket if you want it fixed.
- `flux_calibrate`: output flux unit (docstring says erg/s/cm^2/A but the zeropoint constant is built with a 1e-17 scale); `standard_airmass` has no effect because no extinction curve is passed; exposure times are not read from the Frame Table or headers (default 1.0 s); the engine ignores `order=`.
- `coadd_orders`: `obj_id` is unique across orders, so passing it yields a one-order coadd. Intended?
- `extinction_correct`: dust-map lookup is nearest in squared RA/Dec difference (no cos(dec) term, no angular distance).
- `trace_orders` (ticket 01): origin of `FIRE_ORDER_SPAT_POS` still unverified.
- Parameter keys shown in the YAML blocks are checked by `tests/test_docs.py` only against the string literals in `engines.py`, so the defaults quoted on each page were checked by reading the code, not by a test.
- Steps that do not read `order=` and say so on their pages: `prep_calibrations`, `process_science`, `subtract_sky`, `flux_calibrate`, `telluric_correct`, `coadd_orders`, `extinction_correct`.
