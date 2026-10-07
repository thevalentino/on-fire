"""Fixed-width ASCII Frame Table: the human-editable exposure role table."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from onfire.setup.frames import NUMERIC_FIELDS, Frame


@dataclass(frozen=True)
class Column:
    """A single fixed-width column in the Frame Table."""

    name: str
    width: int
    align: str = "left"
    format_spec: str = ""


COLUMNS: tuple[Column, ...] = (
    Column("idx", 5, "right"),
    Column("filename", 40),
    Column("frame_type", 12),
    Column("object", 28),
    Column("obstype", 12),
    Column("exptime", 9, "right", ".3f"),
    Column("acqtime", 15, "right", ".6f"),
    Column("airmass", 8, "right", ".4f"),
    Column("grism", 6),
    Column("instrume", 8),
)

_NUMERIC_COLUMNS = frozenset(NUMERIC_FIELDS)

_FRAME_FIELDS: dict[str, Callable[[Frame], object]] = {
    "filename": lambda f: f.filename,
    "frame_type": lambda f: f.frame_type,
    "object": lambda f: f.object_name,
    "obstype": lambda f: f.obstype,
    "exptime": lambda f: f.exptime,
    "acqtime": lambda f: f.acqtime,
    "airmass": lambda f: f.airmass,
    "grism": lambda f: f.grism,
    "instrume": lambda f: f.instrume,
}


def _offsets(columns: tuple[Column, ...]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for i, col in enumerate(columns):
        starts.append(pos)
        pos += col.width
        if i != len(columns) - 1:
            pos += 1
    return starts


def _render_text(col: Column, value: object) -> str:
    if value is None:
        text = ""
    elif col.format_spec:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            text = str(value)
        else:
            text = "" if math.isnan(number) else format(number, col.format_spec)
    else:
        text = str(value)

    if len(text) > col.width:
        raise ValueError(
            f"value {text!r} does not fit column {col.name!r} (width {col.width}); "
            "widen the column or shorten the value"
        )
    return text.ljust(col.width) if col.align == "left" else text.rjust(col.width)


def _render_row(index: int, frame: Frame) -> str:
    parts = []
    for col in COLUMNS:
        value = index if col.name == "idx" else _FRAME_FIELDS[col.name](frame)
        parts.append(_render_text(col, value))
    return " ".join(parts)


def _render_header() -> str:
    line = " ".join(_render_text(col, col.name) for col in COLUMNS)
    return "#" + line[1:]


def _parse_value(col: Column, raw: str) -> object:
    raw = raw.strip()
    if col.name == "idx":
        return raw
    if col.name in _NUMERIC_COLUMNS:
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


@dataclass
class FrameTable:
    """An ordered collection of :class:`~onfire.setup.frames.Frame` rows."""

    frames: list[Frame] = field(default_factory=list)
    source: str = ""

    @classmethod
    def from_frames(cls, frames: list[Frame], source: str = "") -> "FrameTable":
        return cls(frames=list(frames), source=source)

    def to_rows(self) -> list[dict[str, str]]:
        """Return the rendered table cells, stripped, keyed by column name."""
        rows = []
        for i, frame in enumerate(self.frames, start=1):
            row: dict[str, str] = {}
            for col in COLUMNS:
                value = i if col.name == "idx" else _FRAME_FIELDS[col.name](frame)
                row[col.name] = _render_text(col, value).strip()
            rows.append(row)
        return rows

    def to_string(self) -> str:
        lines = ["# onfire Frame Table"]
        if self.source:
            lines.append(f"# source: {self.source}")
        lines.append(_render_header())
        for i, frame in enumerate(self.frames, start=1):
            lines.append(_render_row(i, frame))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_string(cls, text: str) -> "FrameTable":
        starts = _offsets(COLUMNS)
        frames: list[Frame] = []
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = {
                col.name: _parse_value(col, line[start : start + col.width])
                for col, start in zip(COLUMNS, starts)
            }
            filename = str(values["filename"])
            frames.append(
                Frame(
                    filename=filename,
                    path=Path(filename),
                    frame_type=str(values["frame_type"]) or "unknown",
                    obstype=str(values["obstype"]),
                    object_name=str(values["object"]),
                    exptime=values["exptime"],  # type: ignore[arg-type]
                    acqtime=values["acqtime"],  # type: ignore[arg-type]
                    airmass=values["airmass"],  # type: ignore[arg-type]
                    grism=str(values["grism"]),
                    instrume=str(values["instrume"]),
                )
            )
        return cls(frames=frames)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_string())
        return path

    @classmethod
    def read(cls, path: str | Path) -> "FrameTable":
        path = Path(path)
        return cls.from_string(path.read_text())
