# 01: Extract Stale marks, provenance, and lineage into small types

**What to build:** A behaviour-preserving prefactor of the State Clone, Commit, and Stale work (see `../spec.md` and ADR 0004). The "Stale for every order" marker, the product-to-step and QA-Artifact-to-step records, and the clone lineage (identity, parent, base revision, revision) each become a small, named concept instead of loose fields and a `None`-means-all-orders convention. The duplicated QA-Artifact pruning, the duplicated "steps recorded so far" computation, and the mirrored checkpoint conversions for Stale marks collapse into one place each. Nothing observable changes: every existing test passes untouched.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The Stale marks, provenance, and lineage are each represented by a dedicated small type; no code outside it depends on the `None`-means-every-order encoding
- [ ] QA-Artifact pruning and the "recorded steps" computation each exist once
- [ ] Checkpoint reading and writing of Stale marks uses one conversion path
- [ ] `ReductionState.commit` no longer assigns snapshot fields one by one
- [ ] The full existing test suite passes with no test changes
- [ ] `GLOSSARY.md` vocabulary is used in names and docstrings (no "promote", "copy", "branch" for Commit/State Clone)
