# 01: Build tooling, core pages, and one model step page

**What to build:** The documentation build and the pages that don't repeat per step (see `../spec.md`). `docs/build.py` renders Markdown from `docs/guide/` into committed HTML under `docs/site/` with the shared template, themes, sidebar, copy buttons, vendored KaTeX, `<details>` sections, figure placeholders, and ast-based code extraction. Write the big-picture page, the quickstart (`onfire setup` then `onfire reduce`, placeholder paths), the Frame Table page, the Parameter Config page, and the interactive-analysis page (State Clone, compare, Commit, Stale). Write one step page in full (suggest `trace_orders`) as the model for the template, including drafted methods text and code snippets. Link the guide from `README.md`. Add the `docs` extra and `tests/test_docs.py`.

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `python docs/build.py` regenerates `docs/site/` and `docs/site/index.html` opens offline with working nav, themes, and KaTeX
- [x] `docs` optional extra (`markdown`, `pygments`) in `pyproject.toml`; not a runtime dependency
- [x] Missing figures render a labelled "figure pending" box; present PNG or Plotly HTML files render for real
- [x] Code snippets are extracted by reference with a file/line label; a bad reference fails the build
- [x] Big-picture, quickstart, Frame Table, Parameter Config, and interactive-analysis pages exist, with examples taken from real code and CLI behaviour
- [x] One complete model step page follows the fixed template, with both `<details>` sections collapsed by default; unconfirmable method claims are marked "to verify"
- [x] The other 11 steps appear in the sidebar (stub or missing-page handling decided and documented in the build script)

## Comments

- Steps without a page show greyed out in the sidebar (`expected_steps` in `docs/build.py`).
- Untested by design (agreed seams): CSS/themes, copy-button JS, KaTeX rendering, prose. Math protection from Markdown (`_stash_math`) has no dedicated test; it was checked only by building the page. Worth confirming equations render in a browser.
- Methods text on `trace_orders` flags one item "to verify" (origin of `FIRE_ORDER_SPAT_POS`).
- No type checker is installed in this environment, so typechecking was not run.
- [x] `tests/test_docs.py` covers flag/step-name existence, reference resolution, and committed output equals fresh build; skipped without the `docs` extra
- [x] Prose uses GLOSSARY.md terms ("Intermediate State" vs "checkpoint file")
- [x] `README.md` links to the guide
