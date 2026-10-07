# 02: Record which orders each product covers

**What to build:** Provenance records, for every product a step produces, which orders it covers, as the spec's Implementation Decisions require, instead of scoping Stale only by step. After an order-targeted re-run on a State Clone, Commit discards only the stale orders from any product that has an order axis (an order-tagged table or an array with an order dimension), and discards products with no order axis whole. Steps that combine orders keep going Stale for every order. The covered orders are derived by the state from the step result and the targeted order; Step Engines stay pure (ADR 0003).

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] A single-order re-run followed by Commit trims only that order from order-axis arrays, not just from tables with an `order` column
- [ ] Products with no order axis are still discarded whole when stale
- [ ] Cross-order steps are still discarded whole when any input order is stale
- [ ] Coverage survives a clone, and a checkpoint save and load
- [ ] Checkpoints written before this change still load, treating their products as whole-state
- [ ] Progress after Commit still reflects the last step with no Stale products
- [ ] Tests drive the public state API with fake Step Engines and assert only externally visible products and progress
