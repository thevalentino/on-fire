# Spec: Static HTML User Guide

Status: ready-for-agent

## Problem Statement

onfire has no user-facing documentation. `README.md` is two lines. An astronomer who knows FIRE data and echelle reduction but is new to onfire cannot learn what the pipeline does at a high level, how to run it, or how each Reduction Step works, without reading source, ADRs, and `.scratch/` specs. The interactive State Clone / Commit workflow in particular is hard to discover.

## Solution

A static, offline-readable HTML guide: a big-picture page, a quickstart, pages for the Frame Table and Parameter Config, an interactive-analysis page, and one page per Reduction Step (12, matching `STEP_SEQUENCE`). Each step page explains how to call the step and holds two collapsed sections: the numerical methods and general idea, and snippets of the pipeline code showing where and how it happens. Figures start as visible placeholders and are populated later from a real data set. Audience: astronomers who know FIRE and echelle reduction. No contributor internals (ADRs and `GLOSSARY.md` cover those).

## Decisions

- **Format and build:** sources are Markdown in `docs/guide/`. `docs/build.py` (tooling, outside the `onfire` package) renders them to `docs/site/` using `markdown` and `pygments`, added as a `docs` optional extra in `pyproject.toml` beside `test`. Not a runtime dependency. No Sphinx or MkDocs.
- **Generated output is committed**, so `docs/site/index.html` opens straight from a clone, offline. KaTeX is vendored into `docs/site/` for equations (no CDN).
- **Look:** one shared stylesheet, light and dark themes following the system setting, a sidebar listing pages with the steps in pipeline order, a copy button on every code block, no JS framework. In-page search is deferred.
- **Step page template, fixed order:** (1) one-line purpose; (2) inputs and outputs (Intermediate State fields read and written); (3) how to call it: CLI (`--until`/`--from`), Parameter Config keys, Python; (4) figure placeholders; (5) collapsed "Numerical methods and idea"; (6) collapsed "Where this happens in the code". Both collapsed sections start hidden (`<details>`).
- **Figures:** declared in page source as `{{figure: <step>/<name> "caption"}}`. If `docs/guide/figures/<step>/<name>.png` (or an embeddable Plotly `.html` QA Artifact) is missing, the build renders a labelled "figure pending" box; if present, the real image or embed. Populating the guide later means dropping files in.
- **Code snippets** are references (file + function), extracted with `ast` at build time and shown with a file path/line label. Never pasted or paraphrased. A renamed or deleted target fails the build.
- **Methods text** is drafted from the code in `onfire/core/`, restricted to what the code does. Anything not confirmable is marked "to verify" instead of filled with textbook assumptions. PypeIt (vendored) may be cross-referenced. The author reviews for scientific correctness.
- **Terminology:** no new glossary terms. Use GLOSSARY.md vocabulary strictly. "Intermediate State" for the concept, "checkpoint file" for the on-disk FITS artifact. A check for GLOSSARY `_Avoid_` words in prose is a nice-to-have, not required.
- **Tests** (`tests/test_docs.py`, `importorskip` on the `docs` extra so the plain `test` install passes): every `onfire` flag and step name mentioned in the pages exists in the real parser / `STEP_SEQUENCE`; every code reference resolves; committed `docs/site/` equals a fresh build.
- **No ADR.** Script generation, extraction by reference, and committed output are cheap to reverse.
- **Examples** come from real signatures and behaviour read from the code, never from memory. Walkthrough paths and outputs are placeholders until real data arrives; no invented numbers or output.

## Out of Scope

- Hosted site, CI deploy, or a static-site framework.
- In-page search.
- Real example data and populated figures (ticket 03).
- Contributor / architecture documentation.

## Tickets

1. `issues/01-build-tooling-core-pages-and-model-step.md`
2. `issues/02-remaining-step-pages.md`
3. `issues/03-populate-real-data-and-figures.md`
