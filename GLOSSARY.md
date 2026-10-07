# Onfire Domain Model

Independent data reduction pipeline for Magellan/FIRE echelle spectroscopy.

## Language

**Order**:
A physically dispersed spectral segment imaged onto the detector, corresponding to an echelle diffraction order (FIRE orders 11 through 31).
_Avoid_: Slit, aperture, track

**Reduction Step**:
A distinct, inspectable stage of data processing that takes an explicit input state and produces an intermediate state and diagnostic artifacts.
_Avoid_: Task, phase, stage, job

**Intermediate State**:
The inspectable in-memory data and on-disk checkpoint produced after executing a reduction step.
_Avoid_: Checkpoint, cache, dump

**State Clone**:
An isolated duplicate of an intermediate state created during interactive analysis to test parameter variations without modifying the baseline.
_Avoid_: Fork, branch, copy

**Baseline**:
The canonical Intermediate State that a State Clone was cloned from and that downstream steps and checkpoints build on.
_Avoid_: Original, master state, main state

**Commit**:
Promoting a State Clone to become the Baseline by replacing it wholesale, discarding all products downstream of the committed step.
_Avoid_: Overwrite, merge, promote, apply

**Stale**:
A mark on an intermediate state's products indicating that an earlier reduction step has been re-run since they were produced, so they no longer match the current calibrations. Stale products are kept until a Commit discards them.
_Avoid_: Dirty, invalid, outdated

**Step Engine**:
A pure functional, array-driven algorithm that executes a specific scientific computation independently of state management.
_Avoid_: Worker, backend, runner

**Cross-Order Regularization**:
Constraining a single-order calibration fit using global trends or adjacent order solutions during interactive single-order execution.
_Avoid_: 2D smoothing, order borrowing

**QA Artifact**:
An interactive Plotly visualization or report generated alongside a reduction step to evaluate processing quality.
_Avoid_: Diagnostic plot, inspection plot

**Pair Differencing**:
Background subtraction performed by differencing two nodded dither exposures (A - B) along the slit.
_Avoid_: A-B subtraction, dither subtraction

**Stare Mode**:
Science exposure observed at a single fixed slit position without spatial nodding, requiring 2D background sky modeling.
_Avoid_: Single frame, non-dithered

**Frame Table**:
A human-readable tabular file with fixed-width columns listing input raw FITS exposures, metadata, and assigned reduction roles.
_Avoid_: File list, manifest, pypeit file, data block

**Parameter Config**:
A YAML configuration file defining pipeline defaults and step-specific algorithmic parameter overrides.
_Avoid_: Control file, par file, settings dump

**Telluric Correction**:
Removal of molecular absorption features imprinted by Earth's atmosphere using standard star observations or atmospheric transmission models.
_Avoid_: Atmospheric de-absorption, telluric cleaning

**Empirical Telluric Correction**:
Atmospheric absorption correction derived from an observed A0V standard star spectrum.
_Avoid_: Star cleaning, standard division

**Grid Telluric Correction**:
Atmospheric absorption correction fitted against a precomputed PCA atmospheric transmission model grid.
_Avoid_: TelFit grid, PCA cleaning

**Dust Extinction Correction**:
Dereddening of top-of-atmosphere fluxes to intrinsic astrophysical values using Galactic dust maps ($E(B-V)$) and an extinction curve.
_Avoid_: Galactic dereddening, dust unmasking

**Order Mask**:
A boolean status flag marking a specific echelle order as unusable or skipped during batch reduction without halting remaining orders.
_Avoid_: Bad order, dead order, skipped order

**Fail-Fast Mode**:
An execution setting that halts on the first step or order failure, saving current state and printing reproducible interactive commands.
_Avoid_: Abort mode, crash mode
