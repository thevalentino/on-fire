"""Build the static HTML user guide from Markdown sources.

Usage: ``python docs/build.py`` renders ``docs/guide/*.md`` into ``docs/site/``.
Requires the ``docs`` extra (``pip install -e .[docs]``).
"""

from __future__ import annotations

import ast
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "guide"
DEFAULT_OUT = HERE / "site"
DEFAULT_ROOT = HERE.parent
STATIC = HERE / "static"

CODE_PATTERN = re.compile(r"\{\{code:\s*(?P<path>[\w./-]+)::(?P<name>[\w.]+)\s*\}\}")
FIGURE_PATTERN = re.compile(r'\{\{figure:\s*(?P<key>[\w./-]+)\s+"(?P<caption>[^"]*)"\s*\}\}')

# Figure file suffix -> embed markup, in lookup order.
FIGURE_EMBEDS = {
    ".png": '<img src="{url}" alt="{caption}">',
    ".html": '<iframe src="{url}" title="{caption}" loading="lazy"></iframe>',
}


def _render_figure(match: re.Match, src: Path, out: Path) -> str:
    key = match["key"]
    caption = html.escape(match["caption"])
    for suffix, template in FIGURE_EMBEDS.items():
        source = src / "figures" / f"{key}{suffix}"
        if source.is_file():
            target = out / "figures" / f"{key}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            url = f"figures/{html.escape(key)}{suffix}"
            return (
                "<figure>"
                + template.format(url=url, caption=caption)
                + f"<figcaption>{caption}</figcaption></figure>"
            )
    return (
        '<figure class="pending">'
        f"<div>figure pending: <code>{html.escape(key)}</code></div>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


# Fixed section order of a step page; the last two start collapsed.
STEP_SECTIONS = (
    "Inputs and outputs",
    "How to call it",
    "Figures",
    "Numerical methods and idea",
    "Where this happens in the code",
)
COLLAPSED_SECTIONS = STEP_SECTIONS[3:]

SECTION_PATTERN = re.compile(r"^## (.+?)\s*$", re.M)
META_PATTERN = re.compile(r"^(\w+):[ \t]*(.*)$")


class StepPageError(Exception):
    """A step page does not follow the fixed section template."""


class CodeReferenceError(Exception):
    """A ``{{code: ...}}`` reference does not resolve to a definition."""


def _find_definition(tree: ast.Module, dotted: str) -> ast.AST | None:
    scope: list[ast.stmt] = tree.body
    node = None
    for part in dotted.split("."):
        node = next(
            (
                n
                for n in scope
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == part
            ),
            None,
        )
        if node is None:
            return None
        scope = node.body
    return node


def _render_code(match: re.Match, root: Path) -> str:
    """Extract a function or class from the source and render it highlighted."""
    rel, name = match["path"], match["name"]
    path = root / rel
    if not path.is_file():
        raise CodeReferenceError(f"code reference {rel}::{name}: no such file {rel}")
    lines = path.read_text().splitlines()
    node = _find_definition(ast.parse("\n".join(lines)), name)
    if node is None:
        raise CodeReferenceError(f"code reference {rel}::{name}: {name} not found in {rel}")
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    source = "\n".join(lines[start - 1 : node.end_lineno])
    body = highlight(source, PythonLexer(), HtmlFormatter(cssclass="highlight"))
    return (
        '<div class="code-ref">'
        f'<div class="code-label">{html.escape(rel)}:{start}</div>{body}</div>'
    )


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split leading ``key: value`` lines (ended by a blank line) from the body."""
    meta: dict[str, str] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (m := META_PATTERN.match(lines[index])):
        meta[m[1]] = m[2]
        index += 1
    if not meta:
        return {}, text
    return meta, "\n".join(lines[index:])


def _md(text: str) -> str:
    return markdown.markdown(text, extensions=["fenced_code", "codehilite"])


def _render_step_page(page: Path, meta: dict[str, str], body: str) -> str:
    """Render a step page, enforcing the fixed sections and collapsing two."""
    headings = SECTION_PATTERN.findall(body)
    if tuple(headings) != STEP_SECTIONS:
        missing = [name for name in STEP_SECTIONS if name not in headings]
        raise StepPageError(
            f"{page.name}: step page sections must be {list(STEP_SECTIONS)} in order; "
            f"found {headings}" + (f" (missing {missing})" if missing else "")
        )
    chunks = SECTION_PATTERN.split(body)[1:]
    parts = [f"<h1>{html.escape(meta.get('title', page.stem))}</h1>"]
    if "purpose" in meta:
        parts.append(f'<p class="purpose">{html.escape(meta["purpose"])}</p>')
    for title, content in zip(chunks[::2], chunks[1::2]):
        rendered = _md(content)
        if title in COLLAPSED_SECTIONS:
            parts.append(f"<details><summary>{title}</summary>{rendered}</details>")
        else:
            parts.append(f"<section><h2>{title}</h2>{rendered}</section>")
    return "\n".join(parts)


MATH_PATTERN = re.compile(
    r"\$\$.+?\$\$|(?<![\\$])\$[^\s$](?:[^$\n]*?[^\s$])?\$", re.S
)
# Fenced blocks and inline code spans, which must never be scanned for math.
CODE_SPAN_PATTERN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.S)

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - onfire</title>
<link rel="stylesheet" href="static/katex/katex.min.css">
<link rel="stylesheet" href="static/pygments.css">
<link rel="stylesheet" href="static/style.css">
</head>
<body>
<nav class="sidebar">
<a class="brand" href="index.html">onfire</a>
{nav}
</nav>
<main>
{body}
</main>
<script defer src="static/katex/katex.min.js"></script>
<script defer src="static/katex/auto-render.min.js"></script>
<script defer src="static/guide.js"></script>
</body>
</html>
"""


@dataclass
class Page:
    """One guide page: where it comes from and what the sidebar needs to know."""

    source: Path
    title: str
    kind: str
    order: int
    body: str = ""

    @property
    def name(self) -> str:
        return self.source.stem


def _stash_math(text: str, store: list[str]) -> str:
    """Replace math outside code with tokens so Markdown cannot mangle it."""

    def stash(match: re.Match) -> str:
        store.append(match[0])
        return f"MATHSNIPPET{len(store) - 1}X"

    pieces = CODE_SPAN_PATTERN.split(text)
    # Odd indices are the code spans captured by the split.
    return "".join(
        piece if index % 2 else MATH_PATTERN.sub(stash, piece)
        for index, piece in enumerate(pieces)
    )


def _title_of(meta: dict[str, str], text: str, fallback: str) -> str:
    if "title" in meta:
        return meta["title"]
    heading = re.search(r"^# (.+?)\s*$", text, re.M)
    return heading[1] if heading else fallback


def _nav(pages: list[Page], current: Page, expected_steps: Sequence[str]) -> str:
    def link(page: Page) -> str:
        here = ' aria-current="page"' if page is current else ""
        return f'<li><a href="{page.name}.html"{here}>{html.escape(page.title)}</a></li>'

    general = sorted((p for p in pages if p.kind != "step"), key=lambda p: (p.order, p.name))
    steps = {p.name: p for p in pages if p.kind == "step"}
    items = [link(p) for p in general]
    parts = ["<ul>" + "".join(items) + "</ul>"]
    step_items = []
    for name in expected_steps:
        if name in steps:
            step_items.append(link(steps[name]))
        else:
            step_items.append(f'<li><span class="todo">{html.escape(name)}</span></li>')
    extras = sorted(n for n in steps if n not in expected_steps)
    step_items.extend(link(steps[n]) for n in extras)
    if step_items:
        parts.append('<div class="group">Reduction steps</div><ul>' + "".join(step_items) + "</ul>")
    return "\n".join(parts)


def _write_static(out: Path) -> None:
    target = out / "static"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(STATIC, target)
    light = HtmlFormatter(style="default").get_style_defs(".highlight")
    dark = HtmlFormatter(style="monokai").get_style_defs(".highlight")
    (target / "pygments.css").write_text(
        f"{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n"
    )


def build(
    src: Path = DEFAULT_SRC,
    out: Path = DEFAULT_OUT,
    root: Path = DEFAULT_ROOT,
    expected_steps: Sequence[str] = (),
) -> None:
    """Render every ``*.md`` page under ``src`` to ``out/<name>.html``.

    ``expected_steps`` lists step names the sidebar shows in order; those without
    a page appear greyed out.
    """
    out.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    for source in sorted(src.glob("*.md")):
        text = FIGURE_PATTERN.sub(lambda m: _render_figure(m, src, out), source.read_text())
        snippets: list[str] = []

        def stash(match: re.Match) -> str:
            snippets.append(_render_code(match, root))
            return f"\n\nCODESNIPPET{len(snippets) - 1}X\n\n"

        text = CODE_PATTERN.sub(stash, text)
        maths: list[str] = []
        text = _stash_math(text, maths)
        meta, text = _split_front_matter(text)
        if meta.get("kind") == "step":
            body = _render_step_page(source, meta, text)
        else:
            body = _md(text)
        for index, snippet in enumerate(snippets):
            body = body.replace(f"<p>CODESNIPPET{index}X</p>", snippet)
        for index, math in enumerate(maths):
            body = body.replace(f"MATHSNIPPET{index}X", html.escape(math, quote=False))
        pages.append(
            Page(
                source=source,
                title=_title_of(meta, text, source.stem),
                kind=meta.get("kind", "page"),
                order=int(meta.get("order", 100)),
                body=body,
            )
        )
    for page in pages:
        shell = SHELL.format(
            title=html.escape(page.title),
            nav=_nav(pages, page, expected_steps),
            body=page.body,
        )
        (out / f"{page.name}.html").write_text(shell)
    _write_static(out)


def main() -> None:
    from onfire.pipeline.steps import STEP_SEQUENCE

    build(expected_steps=STEP_SEQUENCE)


if __name__ == "__main__":
    main()
