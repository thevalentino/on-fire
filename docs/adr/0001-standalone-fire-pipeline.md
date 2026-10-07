# Standalone Pipeline Extracting PypeIt Logic

We decided to build `onfire` as an independent Python package extracting reduction algorithms directly from PypeIt rather than subclassing or wrapping PypeIt as an upstream dependency. This eliminates PypeIt's rigid execution harness to provide full interactive step-level inspectability, state branching, and order-by-order re-execution tailored specifically to Magellan/FIRE echelle spectroscopy, at the cost of manual upkeep for upstream algorithmic improvements.
