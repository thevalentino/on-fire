"""Header ingestion and frame classification for raw Magellan/FIRE exposures."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from astropy.io import fits

# Raw FITS header cards that carry the metadata recorded in the Frame Table.
METADATA_CARDS: dict[str, str] = {
    "obstype": "OBSTYPE",
    "object_name": "OBJECT",
    "exptime": "EXPTIME",
    "acqtime": "ACQTIME",
    "airmass": "AIRMASS",
    "grism": "GRISM",
    "instrume": "INSTRUME",
}

# Frame roles recorded in the Frame Table.
FRAME_TYPES = ("science", "standard", "pixelflat", "arc", "unknown")

# Metadata fields parsed as floating-point numbers rather than strings.
NUMERIC_FIELDS: tuple[str, ...] = ("exptime", "acqtime", "airmass")

# String representations of FIRE ``OBSTYPE`` values grouped by role. Matching is
# case-insensitive and falls back to substring tests for values such as
# ``PixFlat``.
_FLAT_TYPES = {"pixflat", "pixel_flat", "pixelflat", "flat", "domeflat", "internalflat"}
_STANDARD_TYPES = {"telluric", "standard", "std", "spectrophotometric"}
_ARC_TYPES = {"arc", "lamp", "thar", "ar", "ne", "xe", "xenon"}
_SCIENCE_TYPES = {"science", "object", "target"}


@dataclass
class Frame:
    """Metadata and assigned reduction role for a single raw exposure."""

    filename: str
    path: Path
    frame_type: str
    obstype: str = ""
    object_name: str = ""
    exptime: Optional[float] = None
    acqtime: Optional[float] = None
    airmass: Optional[float] = None
    grism: str = ""
    instrume: str = ""
    missing_cards: list[str] = field(default_factory=list)


def classify_frame(obstype: str, object_name: str = "") -> str:
    """Assign a Frame Table role from header metadata.

    Args:
        obstype: Value of the FITS ``OBSTYPE`` card.
        object_name: Value of the FITS ``OBJECT`` card, used as a fallback hint.

    Returns:
        One of :data:`FRAME_TYPES`.
    """
    obstype = (obstype or "").strip().lower()
    object_name = (object_name or "").strip().lower()

    if obstype in _FLAT_TYPES or "flat" in obstype:
        return "pixelflat"
    if obstype in _STANDARD_TYPES or "telluric" in obstype:
        return "standard"
    if obstype in _ARC_TYPES or "arc" in obstype or "lamp" in obstype:
        return "arc"
    if obstype in _SCIENCE_TYPES:
        return "science"

    if "flat" in object_name:
        return "pixelflat"
    if "telluric" in object_name or "standard" in object_name:
        return "standard"
    if "arc" in object_name or "lamp" in object_name:
        return "arc"

    return "unknown"


def _as_float(header: fits.Header, card: str) -> Optional[float]:
    if card not in header:
        return None
    try:
        return float(header[card])
    except (TypeError, ValueError):
        return None


def read_frame(path: str | Path) -> Frame:
    """Read and classify a single raw FITS exposure.

    Missing metadata cards are recorded on the returned :class:`Frame` and a
    warning is emitted so observers are alerted to incomplete headers.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header

    missing = [card for card in METADATA_CARDS.values() if card not in header]
    if missing:
        warnings.warn(
            f"{path.name}: missing header card(s): {', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )

    obstype = str(header.get("OBSTYPE", "") or "")
    object_name = str(header.get("OBJECT", "") or "")

    return Frame(
        filename=path.name,
        path=path,
        frame_type=classify_frame(obstype, object_name),
        obstype=obstype,
        object_name=object_name,
        exptime=_as_float(header, "EXPTIME"),
        acqtime=_as_float(header, "ACQTIME"),
        airmass=_as_float(header, "AIRMASS"),
        grism=str(header.get("GRISM", "") or ""),
        instrume=str(header.get("INSTRUME", "") or ""),
        missing_cards=missing,
    )


def scan_raw_dir(raw_dir: str | Path) -> list[Frame]:
    """Read and classify every ``*.fits`` file directly inside ``raw_dir``.

    Files are returned sorted by filename so Frame Table output is stable.
    A file that cannot be read as FITS is skipped with a clear warning rather
    than aborting the whole scan.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(p for p in raw_dir.glob("*.fits") if p.is_file())

    frames: list[Frame] = []
    for path in paths:
        try:
            frames.append(read_frame(path))
        except Exception as exc:  # noqa: BLE001 - report any unreadable file clearly
            warnings.warn(
                f"{path.name}: could not read as FITS ({exc}); skipping",
                UserWarning,
                stacklevel=2,
            )
    return frames
