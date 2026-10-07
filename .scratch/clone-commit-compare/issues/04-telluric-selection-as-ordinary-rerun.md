# 04: Telluric selection as an ordinary re-run

**What to build:** Choosing between Empirical Telluric Correction and Grid Telluric Correction follows the same workflow as every other step (spec stories 27, 30, and 31; ADR 0004). The `telluric_correct` Step Engine accepts a method parameter, so an observer re-runs the step on a State Clone with `method="grid"` (or `"empirical"`), compares it with the Baseline, and Commits. Telluric registers its own comparison builder so `compare` keeps a purpose-built layout for the two corrections. `compare_telluric` stays as a thin alias that keeps working, with its deprecated `commit=` argument still warning. The change to the engine is limited to method selection; the numerical algorithms for both corrections are untouched (the user explicitly approved this engine change).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] Running `telluric_correct` with a method parameter on a State Clone selects that correction as the canonical telluric products and records the method
- [x] Without the parameter, the automated run's existing selection behaviour is unchanged
- [x] An unknown method is rejected with a clear error listing the available methods
- [x] `compare` on the `telluric_correct` step uses the registered telluric comparison and returns a dual figure for the two states
- [x] Committing the chosen clone makes the Baseline's telluric products and method match it, discarding any products still Stale
- [x] `compare_telluric()` and its `select=` and deprecated `commit=` arguments keep working
- [x] Both telluric corrections produce numerically identical results to before, per the existing telluric tests
