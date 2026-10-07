# Spec: Magellan/FIRE Echelle Data Reduction Pipeline (`onfire`)

Status: ready-for-agent

## Problem Statement

Astronomers reducing near-infrared echelle spectroscopy from the Magellan/FIRE spectrograph currently rely on existing reduction software such as PypeIt or legacy IDL pipelines like FIREHOSE. However, PypeIt's architecture is monolithic, rigid, and tightly coupled to a complex internal container hierarchy and an opaque configuration system. When calibration or extraction steps fail or produce suboptimal results (such as difficult arc line re-identification or tracing errors on faint echelle orders), observers cannot easily inspect the intermediate states of their data, run individual reduction steps in isolation, or tune parameters for a single echelle order (e.g. order 13) within an interactive Python session. Furthermore, PypeIt does not provide interactive diagnostics (like Plotly) for quick visual assessment in modern environments, and conflates exposure file role lists with algorithm parameters in a single, complex configuration file. Observers need a dedicated, streamlined, and inspectable pipeline that extracts proven algorithms without framework lock-in.

## Solution

Build `onfire`: a standalone, streamlined Python data reduction pipeline specifically designed for Magellan/FIRE echelle spectroscopy (orders 11–31). `onfire` extracts proven numerical reduction algorithms from PypeIt into pure functional step engines, separating mathematical computation from state management and persistence. The pipeline supports both full automated batch processing and granular interactive inspection in IPython/Jupyter sessions. Observers can run individual steps on an inspectable state container, create isolated state clones to test and compare parameter tweaks side-by-side, re-run calibrations on specific individual orders with cross-order regularization, and examine interactive Plotly QA artifacts saved as standalone HTML files or live in-memory figures. Configuration is cleanly divided into a human-readable, fixed-width ASCII Frame Table and a transparent YAML Parameter Config.

## User Stories

1. As an astronomer, I want to scan a raw data directory and generate a fixed-width ASCII Frame Table, so that all raw Magellan/FIRE exposures are automatically categorized with their exposure metadata and reduction roles.
2. As an astronomer, I want the table generation tool to validate input frames, so that I am immediately alerted if required flat or arc calibrations are missing.
3. As an astronomer, I want to edit the fixed-width Frame Table in any standard text editor, so that I can easily reassign frame roles or associate specific dither pairs without syntax errors.
4. As an astronomer, I want a clean YAML configuration file for algorithmic parameter overrides, so that I do not have to navigate deeply nested, instrument-agnostic parameter hierarchies.
5. As an astronomer, I want to run the entire reduction pipeline end-to-end with a single CLI command (`onfire reduce config.yaml`), so that I can obtain fully reduced 1D and 2D science spectra without manual intervention.
6. As an astronomer, I want to halt an automated reduction at a specific reduction step (`--until <step>`), so that I can inspect the intermediate output before proceeding to downstream processing.
7. As an astronomer, I want to resume an automated reduction from a previously saved intermediate checkpoint (`--from <step>`), so that I do not waste time recomputing earlier calibration stages.
8. As an astronomer, I want the pipeline to generate interactive Plotly QA HTML files on disk for each reduction step, so that I can pan, zoom, and visually assess trace quality and residuals in a web browser.
9. As an astronomer, I want to open an interactive IPython session and load any intermediate FITS checkpoint into a `ReductionState`, so that I have immediate access to all underlying arrays, variance maps, and metadata tables.
10. As an astronomer, I want to execute a specific reduction step directly on the `ReductionState` in Python, so that I can interactively inspect the outputs of individual algorithms like tilt fitting or object finding.
11. As an astronomer, I want to execute a reduction step for a single specific echelle order (such as order 13), so that I can debug difficult orders without recomputing the entire detector.
12. As an astronomer, I want to create a State Clone (`clone=True`) when running an interactive step, so that I can test different fitting parameters without mutating my baseline reduction state.
13. As an astronomer, I want to compare the diagnostic plots and extracted outputs of two different State Clones side-by-side, so that I can evaluate which parameter choices yield the highest signal-to-noise ratio.
14. As an astronomer, I want to overwrite the baseline state or save a State Clone as a new checkpoint, so that I have complete control over whether experimental parameter changes become canonical.
15. As an astronomer, I want the pipeline to support pair differencing (A - B nodding), so that thermal infrared sky background and OH airglow lines are subtracted at the exposure level.
16. As an astronomer, I want the pipeline to support stare mode observations, so that single non-dithered science exposures can be reduced using 2D spatial background modeling.
17. As an astronomer, I want to choose between pair differencing and stare mode in the Parameter Config, so that the processing workflow matches my observing strategy.
18. As an astronomer, I want the pipeline to identify arc and OH sky lines and calibrate wavelengths across all 21 orders (orders 11–31) using bundled Magellan/FIRE reference templates, so that the pipeline runs reliably without external downloads.
19. As an astronomer, I want to derive spectral line tilts across each order and apply cross-order regularization during single-order interactive runs, so that noisy or faint orders benefit from the global instrument solution.
20. As an astronomer, I want to perform boxcar and optimal (Horne) 1D spectral extraction per order, so that I obtain maximal signal-to-noise spectra with propagated variance estimates.
21. As an astronomer, I want to perform per-order flux calibration using an observed spectrophotometric standard star, so that raw counts are converted to physical flux density.
22. As an astronomer, I want to perform empirical telluric correction using an observed A0V standard star, so that Earth's atmospheric absorption features are accurately removed.
23. As an astronomer, I want to perform grid-based telluric correction using PCA atmospheric transmission models (TelPCA), so that science targets without nearby standard stars can still be corrected.
24. As an astronomer, I want to run both empirical and grid telluric corrections interactively and compare them in a dual Plotly figure, so that I can choose the cleanest telluric correction for my target.
25. As an astronomer, I want to coadd overlapping calibrated orders 11 through 31 into a continuous 1D spectrum, so that I have a unified spectral product spanning 0.8 to 2.5 microns.
26. As an astronomer, I want to correct the coadded spectrum for Milky Way dust extinction using coordinates from the FITS header, so that I obtain intrinsic astrophysical fluxes.
27. As an astronomer, I want the pipeline to run with `--fail-fast` and display the exact Python code needed to reload the state and re-run the failed step with identical arguments, so that debugging algorithmic failures in IPython is instantaneous.
28. As an astronomer, I want default batch runs to mask problematic orders with an Order Mask and continue reducing healthy orders, so that a failure on one order does not abort an entire reduction sequence.

## Implementation Decisions

### 1. Two-Tier Architecture and Standalone Package
- The codebase lives in `onfire/` as an independent Python package with zero runtime dependency on PypeIt, following [ADR 0001](docs/adr/0001-standalone-fire-pipeline.md).
- Computational algorithms are organized in `onfire.core` as pure functional Step Engines that operate purely on NumPy arrays, Astropy tables, and numeric dictionaries, decoupled from state management ([ADR 0003](docs/adr/0003-two-tier-computational-architecture.md)).
- State management, FITS I/O, and interactive APIs are encapsulated in `onfire.pipeline.state.ReductionState`.

### 2. Configuration Decoupling
- Input frame role associations and metadata are maintained in a human-readable, fixed-width ASCII table (`frames.ascii`), generated and verified by `onfire setup <raw_dir>` ([ADR 0002](docs/adr/0002-separated-configuration-files.md)).
- Algorithmic parameters and pipeline control options are maintained in a YAML file (`config.yaml`).

### 3. Canonical 12-Step Reduction Sequence
The pipeline executes 12 explicit Reduction Steps in sequence:
1. `prep_calibrations`: Combine flat-field frames and construct bad pixel masks.
2. `trace_orders`: Locate and trace the physical spatial boundaries of orders 11 through 31.
3. `calibrate_wavelengths`: Identify arc/OH emission lines and determine 1D wavelength solutions per order.
4. `fit_tilts`: Model the 2D tilt of spectral lines across each order slit.
5. `process_science`: Flat-field science frames and prepare background subtraction via pair differencing (A - B) or stare mode.
6. `find_objects`: Detect spatial object profiles and trace centroids within each order.
7. `subtract_sky`: Perform global and local 2D B-spline sky modeling.
8. `extract_1d`: Perform boxcar and Horne optimal spectral extraction per order.
9. `flux_calibrate`: Determine and apply per-order sensitivity functions from standard star exposures.
10. `telluric_correct`: Remove atmospheric absorption per order via empirical A0V standard star modeling or TelPCA grid fitting.
11. `coadd_orders`: Splicing and weighted combination of overlapping orders into a continuous 1D spectrum.
12. `extinction_correct`: Correct intrinsic fluxes for Galactic dust extinction using $E(B-V)$ maps and Fitzpatrick 1999 extinction law.

### 4. Intermediate State Persistence and Cloning
- Checkpoint files are saved as Multi-Extension FITS files in a flat directory structure prefixed with chronological step numbers (e.g. `01_order_traces.fits`, `02_wavelengths.fits`, ..., `12_final_spectrum.fits`).
- State Clones created via `clone=True` remain in memory during interactive sessions until explicitly saved to disk or committed over the baseline.
- Interactive step executions support order targeting (`order=13`) and incorporate cross-order regularization from the global solution.

### 5. Diagnostics and QA Artifacts
- Every step provides interactive Plotly visualizations when `qa=True`.
- Standalone self-contained HTML files are automatically saved to `qa/<step>_qa.html`, and Plotly `Figure` objects are attached to `state.plots`.

### 6. Error Handling and Fail-Fast
- When `--fail-fast` is enabled, an unhandled error or fit failure halts execution immediately, checkpoints state up to the failure point, and logs the exact Python code required to load the state and re-execute the failed step with its exact parameters.
- In normal batch mode, an order failure sets an `Order Mask` flag on that order and continues processing the remaining orders.

### 7. Bundled Reference Data
- Static Magellan/FIRE optical definitions, `ord_spat_pos` order locations, wavelength templates (`magellan_fire_echelle.fits`), OH line lists, and Vega templates are bundled inside package data with optional path overrides in YAML.

## Testing Decisions

### 1. External Behavior Testing Seam
- Tests will strictly target the highest public seam: `ReductionState` methods (`ReductionState.run_step(...)`, `ReductionState.load(...)`, `ReductionState.save(...)`) and the top-level orchestration runner `onfire.reduce(...)`.
- Tests will verify external contracts: valid output arrays, error propagation, FITS structure compliance, state mutation/cloning behavior, and QA figure generation, without asserting against private internal helper implementations.

### 2. Modules to Test
- `onfire.pipeline.state`: State initialization, cloning, in-place updates, FITS loading and writing.
- `onfire.setup`: Fixed-width table generation, header extraction, and calibration validation.
- `onfire.pipeline.runner`: CLI flags (`--until`, `--from`, `--fail-fast`), step progression, and error reporting.
- `onfire.core.*`: Scientific Step Engines verified through synthetic inputs and regression data for each of the 12 steps.

### 3. Prior Art & Reference Data
- Algorithmic outputs will be cross-validated against reference outputs from PypeIt Magellan/FIRE test suites and canonical FIRE data.

## Out of Scope

- Magellan/FIRE Longslit and PRISM modes (strictly echelle mode, orders 11–31, is supported).
- Spectrographs other than Magellan/FIRE.
- Native desktop graphical user interfaces (all visualization is handled via interactive Plotly HTML and IPython).
- Two-way synchronization or upstream patching of the reference PypeIt repository.

## Further Notes

- Architecture decisions are recorded in:
  - [ADR 0001: Standalone Pipeline Extracting PypeIt Logic](docs/adr/0001-standalone-fire-pipeline.md)
  - [ADR 0002: Separated Configuration Files](docs/adr/0002-separated-configuration-files.md)
  - [ADR 0003: Two-Tier Computational Architecture](docs/adr/0003-two-tier-computational-architecture.md)
- Domain terminology is maintained in [GLOSSARY.md](GLOSSARY.md).
