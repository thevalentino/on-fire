# Spec: Interactive State Clone Commit, Stale Tracking, and Comparison

Status: ready-for-agent

## Problem Statement

An astronomer tuning a reduction interactively can already run a step on an isolated State Clone, but the clone is a dead end. There is no way to adopt a clone that turned out better as the Baseline, so the user must redo the work on the Baseline by hand. Re-running an earlier step (say, wavelength calibration of order 13) leaves later products from the old calibration sitting beside the new one with nothing to say they no longer match. The only side-by-side comparison is telluric-specific, and it uses the word "commit" to mean something else (choosing a telluric method). A previously masked order can be re-run successfully and still stay masked. The result is that the interactive workflow the pipeline exists to provide stops short of letting an observer experiment, compare, and adopt a result safely.

## Solution

The astronomer clones the Baseline, re-runs one or more steps (optionally targeting a single order) on the clone, compares the clone against the Baseline or another clone in a single dual Plotly figure for any step, and, if satisfied, commits the clone. Commit replaces the Baseline wholesale and in place, drops only the products that are still Stale (made out of date by a re-run of an earlier step), and leaves the user to run forward from there. Nothing on disk changes until the user explicitly saves. A clone whose Baseline has moved on since it was cloned is refused unless the user forces it. Successfully re-running a masked order clears its Order Mask.

## User Stories

1. As an astronomer, I want to re-run a calibration step on a State Clone, so that I can try different parameters without risking my Baseline.
2. As an astronomer, I want to target a single order (e.g. `order=13`) when re-running on a clone, so that I only redo the order that failed.
3. As an astronomer, I want to commit a clone that came out better, so that the Baseline adopts my improved result without redoing the work.
4. As an astronomer, I want Commit to mutate the Baseline object I already hold, so that my existing variables and references stay valid.
5. As an astronomer, I want Commit to leave the clone object unchanged, so that I can still compare against it, or commit it again with `force=True` once the Baseline has moved on.
6. As an astronomer, I want Commit to be in-memory only, so that nothing on disk changes until I explicitly call `save`.
7. As an astronomer, I want to save a clone as a new checkpoint without committing it, so that I can keep an experiment without making it canonical.
8. As an astronomer, I want products that came from steps after a re-run step to be marked Stale rather than deleted, so that I can still compare the clone's old downstream outputs.
9. As an astronomer, I want running a step to clear that step's own Stale mark, so that a freshly re-run chain (e.g. wavelength, tilts, extract) is recognized as up to date.
10. As an astronomer, I want Commit to discard exactly the products that are still Stale, so that I never end up with science products built on a calibration that has changed.
11. As an astronomer, I want progress after a Commit to sit at the last non-Stale step, so that I know where to run forward from.
12. As an astronomer, I want a single-order re-run to mark only that order's downstream products Stale, so that I don't have to re-reduce orders that were fine.
13. As an astronomer, I want steps that combine orders (such as order coaddition) to be marked Stale whenever any of their input orders is Stale, so that a spliced spectrum is never built from mismatched orders.
14. As an astronomer, I want products with no order axis to be treated as whole-state for staleness, so that nothing is silently left behind.
15. As an astronomer, I want Commit to refuse a clone whose Baseline has changed since it was cloned, so that I can't silently resurrect stale calibrations.
16. As an astronomer, I want to override that refusal with `force=True`, so that I can still commit when I know the divergence doesn't matter.
17. As an astronomer, I want a direct (non-clone) `run_step` on the Baseline to count as a change to the Baseline, so that clones taken before it are correctly seen as diverged.
18. As an astronomer, I want a clone of a freshly loaded checkpoint to be committable back to that loaded state, so that the load-clone-commit loop works across sessions' fresh starts.
19. As an astronomer, I want two separately loaded states to be treated as separate lineages, so that I can't accidentally commit a clone into an unrelated state.
20. As an astronomer, I want a successful targeted re-run of a masked order to clear its Order Mask, so that Commit carries the recovered order into the Baseline.
21. As an astronomer, I want a failed re-run of a masked order to keep the mask and update its reason, so that the record reflects the latest failure.
22. As an astronomer, I want Commit to carry the clone's Order Mask along with everything else, so that the Baseline never disagrees with the products it holds.
23. As an astronomer, I want one `compare(other, step)` method, so that I can compare two states for any step with the same call.
24. As an astronomer, I want `compare` to return a dual Plotly figure built from each state's QA Artifact for that step, so that I can judge the two side by side.
25. As an astronomer, I want `compare` to raise a clear error naming which state lacks the QA Artifact, so that I know to re-run the step with QA on.
26. As an astronomer, I want that error to mention when a single-order clone only carries that order's artifact, so that a missing-looking figure isn't mysterious.
27. As an astronomer, I want a step to be able to override the comparison figure, so that bespoke comparisons such as telluric keep their purpose-built layout.
28. As an astronomer, I want to compare a clone with the Baseline, or two clones with each other, so that I can rank several experiments.
29. As an astronomer, I want the comparison to be figure-only, so that I get a simple, predictable result without per-step metrics definitions.
30. As an astronomer, I want to choose between empirical and grid telluric correction by re-running the `telluric` step on a clone with a method parameter and committing, so that telluric selection follows the same workflow as every other step.
31. As an astronomer, I want the existing `compare_telluric` call to keep working, with its `commit` argument renamed to `select` and the old name deprecated, so that existing notebooks don't break without warning and the word Commit means one thing.
32. As an astronomer, I want every Stale mark and product-to-step record to survive a Baseline-to-clone copy, so that a clone behaves exactly like the state it came from.
33. As an astronomer, I want these operations only in the Python API, so that the CLI stays a simple batch reduction whose Fail-Fast Mode points me into an interactive session.
34. As an astronomer, I want Plotly figures for steps that Commit discards to be dropped with them, so that the Baseline never displays a QA Artifact for a product it no longer has.
35. As an astronomer, I want checkpoints saved after a Commit to contain no Stale products, so that a later load can't reintroduce mismatched data.

## Implementation Decisions

- **Terms** follow `GLOSSARY.md`: Baseline, State Clone, Commit, Stale, Order Mask, QA Artifact, Intermediate State. "Revision" is a session-local implementation detail and not a glossary term.
- **Governing ADR:** `docs/adr/0004-whole-state-commit-with-deferred-invalidation.md`. Commit is whole-state replacement. There is no per-order or per-step merge between states.
- **Module touched:** the pipeline state container. The Step Engine tier stays pure and unchanged, per ADR-0003. Engines never see staleness, lineage, or commits.
- **Product provenance:** when a step runs, the state records which step produced each product and, where the product has an order axis or column, which orders it covers. This is derived from the step result and the targeted order. Products with no order axis are whole-state.
- **Stale marks:** running step N marks products of all later steps Stale, scoped to the targeted order for a single-order run and to all orders otherwise. Steps that combine orders are Stale if any input order is Stale. Running a step clears the Stale mark on its own products for the orders it covered. Marks are per-state data, copied by cloning.
- **Commit:** `baseline.commit(clone, force=False)`. It replaces the Baseline's Frame Table, Parameter Config, products, metadata, plots and Order Mask with the clone's, drops products still Stale (and their plots), and sets the progress step to the last step with no Stale products. It mutates the Baseline in place, leaves the clone unchanged, and touches no files.
- **Lineage:** each state holds a session-local revision counter. A clone remembers the Baseline's revision at clone time. The counter increments on Commit and on any direct (non-clone) `run_step` on the Baseline. Commit raises a clear error when a clone's remembered revision no longer matches, unless `force=True`. The counter is not written to checkpoints, so each loaded state is its own lineage, and a clone can only be committed back to the in-memory Baseline it came from.
- **Progress rule:** the existing high-water-mark behavior on re-run is kept until Commit. Re-running an earlier step does not rewind progress on a state. Only Commit lowers it.
- **Order Mask on re-run:** a successful run targeting a masked order clears its mask. A failed run keeps the mask and updates its reason.
- **Compare:** `state.compare(other, step)` returns a dual Plotly figure built from the two states' QA Artifacts for that step, raising with a clear error naming the state that lacks one (and noting when a single-order clone carries only one order's artifact). It returns the figure only, with no metrics table. A step may register its own comparison builder to override the default.
- **Telluric:** `compare_telluric` becomes the telluric step's registered comparison builder. Selecting empirical or grid is an ordinary re-run of the `telluric` step with a method parameter, followed by Commit. The old `commit=` argument is renamed `select=`, with the old name accepted and warned as deprecated, so existing notebooks fail loudly, not silently.
- **CLI:** no new subcommands. Commit and compare are Python-API only. The batch runner and Fail-Fast Mode are unchanged.
- **Checkpoints:** the FITS checkpoint format records product provenance and Stale marks so a saved clone round-trips. It does not record the revision counter.

## Testing Decisions

- **One seam:** the public state API: `run_step` (with `clone=True` and `order=N`), `commit`, `compare`, `save` and `load`. Tests register fake Step Engines, so staleness, masks, lineage and figures are observed purely through that API. No test reaches into the provenance record or the revision counter directly.
- **A good test** asserts external behavior: which products exist after Commit, what progress reports, whether the Baseline object identity is preserved, whether the clone is unchanged, whether a save/load round-trip is lossless, which error is raised and what it names. It never asserts how provenance or Stale marks are stored.
- **Cases to cover:** commit replaces the Baseline in place and leaves the clone intact; Commit drops only still-Stale products and rewinds progress to the last non-Stale step; a re-run chain (wavelength, tilts, extract) keeps its refreshed products and drops the unrefreshed later ones; single-order staleness leaves other orders untouched while cross-order steps go Stale; diverged-Baseline refusal and the `force=True` override; a direct Baseline run invalidating earlier clones; separately loaded states being unrelated lineages; mask clearing on success and mask-with-updated-reason on failure; `compare` for two clones, for a clone against the Baseline, for a missing QA Artifact, and for a step with a registered override; the `compare_telluric` alias and `select=` rename with the deprecation warning; nothing written to disk by Commit; saved post-Commit checkpoints containing no Stale products.
- **Prior art:** `tests/test_pipeline_state.py` already exercises clone isolation, single-order forwarding, progress preservation on re-run, checkpoint round-trips and QA attachment with fake engines and the `make_frame` fixture. The new tests follow the same pattern. Telluric comparison coverage lives in `tests/test_flux_telluric.py`.

## Out of Scope

- Per-order or per-step merging of one state into another.
- Per-step metrics tables (median S/N and similar) in `compare`.
- Commit and compare as CLI subcommands, or any GUI.
- Persisting lineage or revision across save and load.
- Automatically re-running Stale steps after a Commit. The user runs forward.
- Changes to any Step Engine's numerical algorithm.

## Further Notes

- The user's mental model is "like a branch in git", but the glossary lists "branch" under State Clone's avoid-list, and this spec keeps the vocabulary as Baseline, State Clone and Commit.
- The Baseline's existing "keep the high-water mark" rule on re-run is deliberate and stays. Stale marks carry the staleness information that the progress counter cannot.
- Provenance is the largest structural change. It is why products without a clear order axis are conservatively treated as whole-state.
