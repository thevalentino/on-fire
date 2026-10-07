# Two-Tier Computational Architecture

We decided to partition the codebase into two strict tiers: pure functional array-based Step Engines in `onfire.core`, and lightweight dataclass state containers in `onfire.pipeline.state`. This isolates numerical reduction algorithms from container and persistence logic, allowing step engines to be called on raw arrays without container overhead while giving interactive users clean state objects with cloning and FITS checkpointing.
