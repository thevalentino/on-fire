"""Tests for the static HTML user guide build (``docs/build.py``)."""

import html as html_lib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("markdown")
pytest.importorskip("pygments")

REPO = Path(__file__).resolve().parents[1]


def _load_build():
    if "docs_build" in sys.modules:
        return sys.modules["docs_build"]
    spec = importlib.util.spec_from_file_location("docs_build", REPO / "docs" / "build.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def visible_text(markup):
    """Return the text a reader sees, with tags stripped and entities decoded."""
    return html_lib.unescape(re.sub(r"<[^>]+>", "", markup))


@pytest.fixture
def guide(tmp_path):
    """Return a factory that builds a fixture guide and returns the output dir."""
    build = _load_build()

    def _build(pages, figures=None, root=None):
        src = tmp_path / "guide"
        out = tmp_path / "site"
        src.mkdir()
        for name, text in pages.items():
            (src / f"{name}.md").write_text(text)
        for rel, data in (figures or {}).items():
            path = src / "figures" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if root is None:
            build.build(src, out)
        else:
            build.build(src, out, root=root)
        return out

    return _build


def test_missing_figure_renders_labelled_pending_box(guide):
    out = guide({"page": '# Page\n\n{{figure: trace_orders/edge-fit "Order edges on the flat"}}\n'})

    html = (out / "page.html").read_text()

    assert "figure pending" in html
    assert "trace_orders/edge-fit" in html
    assert "Order edges on the flat" in html
    assert "<img" not in html


def test_present_png_renders_image_and_is_copied_to_site(guide):
    png = b"\x89PNG fake"
    out = guide(
        {"page": '{{figure: trace_orders/edge-fit "Order edges on the flat"}}\n'},
        figures={"trace_orders/edge-fit.png": png},
    )

    html = (out / "page.html").read_text()

    assert 'src="figures/trace_orders/edge-fit.png"' in html
    assert "Order edges on the flat" in html
    assert "figure pending" not in html
    assert (out / "figures" / "trace_orders" / "edge-fit.png").read_bytes() == png


def test_present_plotly_html_is_embedded_and_copied_to_site(guide):
    plot = b"<html><body>plotly</body></html>"
    out = guide(
        {"page": '{{figure: trace_orders/qa "Interactive QA"}}\n'},
        figures={"trace_orders/qa.html": plot},
    )

    html = (out / "page.html").read_text()

    assert '<iframe src="figures/trace_orders/qa.html"' in html
    assert "Interactive QA" in html
    assert "figure pending" not in html
    assert (out / "figures" / "trace_orders" / "qa.html").read_bytes() == plot


SAMPLE_MODULE = """import math


def helper():
    return 1


def fit_edges(flat):
    \"\"\"Fit the order edges.\"\"\"
    return math.sqrt(flat)
"""


def test_code_reference_shows_real_source_with_path_and_line_label(guide, tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text(SAMPLE_MODULE)

    out = guide({"page": "{{code: pkg/mod.py::fit_edges}}\n"}, root=root)

    html = (out / "page.html").read_text()

    text = visible_text(html)
    assert "return math.sqrt(flat)" in text
    assert "pkg/mod.py:8" in text
    assert "def helper" not in text


def test_code_reference_to_missing_function_fails_the_build(guide, tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text(SAMPLE_MODULE)

    with pytest.raises(_load_build().CodeReferenceError, match="renamed_away"):
        guide({"page": "{{code: pkg/mod.py::renamed_away}}\n"}, root=root)


STEP_PAGE = """kind: step
title: Trace orders
purpose: Locate the edges of every echelle order on the flat.

## Inputs and outputs

Reads the flat.

## How to call it

Run `onfire reduce config.yaml --until trace_orders`.

## Figures

{{figure: trace_orders/edge-fit "Order edges"}}

## Numerical methods and idea

Edges are fit with a polynomial.

## Where this happens in the code

Shown below.
"""


def test_step_page_collapses_methods_and_code_sections_by_default(guide):
    out = guide({"trace_orders": STEP_PAGE})

    html = (out / "trace_orders.html").read_text()

    details = re.findall(r"<details(\s[^>]*)?>\s*<summary>(.*?)</summary>", html, re.S)
    assert [summary for _, summary in details] == [
        "Numerical methods and idea",
        "Where this happens in the code",
    ]
    assert all("open" not in (attrs or "") for attrs, _ in details)
    assert "Locate the edges of every echelle order on the flat." in visible_text(html)


def test_step_page_lists_sections_in_the_fixed_order(guide):
    out = guide({"trace_orders": STEP_PAGE})

    html = (out / "trace_orders.html").read_text()

    headings = re.findall(r"<h2[^>]*>(.*?)</h2>|<summary>(.*?)</summary>", html, re.S)
    assert [a or b for a, b in headings] == [
        "Inputs and outputs",
        "How to call it",
        "Figures",
        "Numerical methods and idea",
        "Where this happens in the code",
    ]


def test_step_page_missing_a_section_fails_the_build(guide):
    broken = STEP_PAGE.replace("## Figures\n\n{{figure: trace_orders/edge-fit \"Order edges\"}}\n\n", "")

    with pytest.raises(_load_build().StepPageError, match="Figures"):
        guide({"trace_orders": broken})


# --- The real guide must agree with the real CLI and step registry --------------

GUIDE = REPO / "docs" / "guide"
SITE = REPO / "docs" / "site"


def _guide_text():
    return {page.name: page.read_text() for page in sorted(GUIDE.glob("*.md"))}


def _cli_options():
    from onfire.cli import build_parser

    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.choices and a.dest == "command")
    options = {}
    for command, sub in subparsers.choices.items():
        options[command] = {o for a in sub._actions for o in a.option_strings}
    return options


def test_every_onfire_flag_in_the_guide_exists_in_the_cli():
    options = _cli_options()
    known = set().union(*options.values()) | {"-h", "--help"}
    problems = []
    for name, text in _guide_text().items():
        for flag in re.findall(r"(?<![\w-])(--?[A-Za-z][\w-]*)", text):
            if flag.startswith("--") and flag not in known and _looks_like_cli(flag, text):
                problems.append(f"{name}: {flag}")
    assert not problems


def _looks_like_cli(flag, text):
    """Only flags that appear on an ``onfire`` command line or in an options table."""
    for line in text.splitlines():
        if flag in line and ("onfire " in line or line.lstrip().startswith("| `-")):
            return True
    return False


def test_every_onfire_subcommand_in_the_guide_exists_in_the_cli():
    commands = set(_cli_options())
    problems = []
    for name, text in _guide_text().items():
        code = "\n".join(re.findall(r"```\w*\n(.*?)```", text, re.S))
        for command in re.findall(r"^\s*(?:\$ )?onfire\s+([a-z]+)", code, re.M):
            if command not in commands:
                problems.append(f"{name}: onfire {command}")
    assert not problems


STEP_REFERENCES = [
    r"--until\s+(\w+)",
    r"--from\s+(\w+)",
    r"run_step\(\s*\"(\w+)\"",
    r"resolve_call_params\(\s*\"(\w+)\"",
    r"compare\([^)]*,\s*\"(\w+)\"\)",
]


def test_every_step_name_the_guide_uses_is_a_real_step():
    from onfire.pipeline.steps import STEP_SEQUENCE

    problems = []
    for name, text in _guide_text().items():
        for pattern in STEP_REFERENCES:
            for step in re.findall(pattern, text):
                if step not in STEP_SEQUENCE and not step.isdigit():
                    problems.append(f"{name}: {step}")
        if re.search(r"^kind:\s*step", text, re.M) and Path(name).stem not in STEP_SEQUENCE:
            problems.append(f"{name}: step page is not a real step")
    assert not problems


def test_overview_lists_the_steps_in_pipeline_order():
    from onfire.pipeline.steps import STEP_SEQUENCE

    listed = re.findall(r"^\d+\. `(\w+)`", _guide_text()["index.md"], re.M)

    assert tuple(listed) == STEP_SEQUENCE


def test_committed_site_matches_a_fresh_build(tmp_path):
    from onfire.pipeline.steps import STEP_SEQUENCE

    _load_build().build(GUIDE, tmp_path, expected_steps=STEP_SEQUENCE)

    fresh = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    committed = {p.relative_to(SITE): p.read_bytes() for p in SITE.rglob("*") if p.is_file()}
    assert committed == fresh, "docs/site is stale; run `python docs/build.py`"


def test_every_reduction_step_has_a_guide_page():
    from onfire.pipeline.steps import STEP_SEQUENCE

    missing = [step for step in STEP_SEQUENCE if not (GUIDE / f"{step}.md").is_file()]

    assert not missing


def test_step_parameters_shown_in_the_guide_are_read_by_the_engines():
    import yaml

    engines = (REPO / "onfire" / "core" / "engines.py").read_text()
    problems = []
    for name, text in _guide_text().items():
        for block in re.findall(r"```yaml\n(.*?)```", text, re.S):
            steps = (yaml.safe_load(block) or {}).get("steps", {})
            for step, params in steps.items():
                for key in params or {}:
                    if f'"{key}"' not in engines and f"'{key}'" not in engines:
                        problems.append(f"{name}: steps.{step}.{key}")
    assert not problems
