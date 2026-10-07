"""Raw exposure ingestion by Frame Table role.

Reading raw FITS files belongs to the persistence tier; the resulting NumPy
stacks are handed to the pure Step Engines through the state products
dictionary (ADR 0003).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onfire.pipeline.state import ReductionState
    from onfire.setup.frames import Frame

ROLE_PRODUCTS: dict[str, str] = {
    "pixelflat": "raw_flats",
    "arc": "raw_arcs",
    "science": "raw_science",
    "standard": "raw_standard",
}

# FITS header cards carrying the telescope pointing, in degrees.
_COORDINATE_CARDS: tuple[tuple[str, str], ...] = (
    ("RA", "DEC"),
    ("RA_DEG", "DEC_DEG"),
    ("TELRA", "TELDEC"),
    ("OBJRA", "OBJDEC"),
)


def read_fits_data(path: str | Path) -> np.ndarray:
    """Read the primary image data of a raw FITS exposure as float."""
    with fits.open(Path(path), memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"{path}: primary HDU contains no image data")
    return np.asarray(data, dtype=float)


def _resolve_path(state: "ReductionState", frame: "Frame") -> Path:
    """Resolve a Frame Table exposure path relative to the Frame Table file."""
    path = Path(frame.path)
    if path.is_absolute() or path.exists():
        return path
    source = getattr(state, "source", "") or getattr(state.frames, "source", "")
    if source:
        candidate = Path(source).parent / path
        if candidate.exists():
            return candidate
    return path


def read_raw_frames(
    state: "ReductionState",
    roles: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read every Frame Table exposure for ``roles`` into a stacked array.

    Args:
        state: The state whose Frame Table lists the raw exposures.
        roles: Frame Table roles to load. Defaults to every known role; roles
            absent from the Frame Table are skipped.

    Returns:
        A mapping of product key to an ``(nimages, nspec, nspat)`` array for
        each role that had at least one frame.
    """
    selected = tuple(roles) if roles is not None else tuple(ROLE_PRODUCTS)
    loaded: dict[str, np.ndarray] = {}
    for role in selected:
        frames = [frame for frame in state.frames.frames if frame.frame_type == role]
        if not frames:
            continue
        stack = np.stack([read_fits_data(_resolve_path(state, frame)) for frame in frames])
        key = ROLE_PRODUCTS.get(role, f"raw_{role}")
        state.products[key] = stack
        loaded[key] = stack
    return loaded


def read_target_coordinates(
    state: "ReductionState",
) -> tuple[float, float] | None:
    """Return the ``(ra, dec)`` target coordinates from the first science frame.

    The pointing is read from the first usable ``RA``/``DEC``-style header card
    on the first science exposure, in degrees. ``None`` is returned when no
    science frame or no coordinate card is present.
    """
    for frame in state.frames.frames:
        if frame.frame_type != "science":
            continue
        path = _resolve_path(state, frame)
        try:
            with fits.open(path, memmap=False) as hdul:
                header = hdul[0].header
        except OSError:
            continue
        for ra_card, dec_card in _COORDINATE_CARDS:
            if ra_card in header and dec_card in header:
                try:
                    return float(header[ra_card]), float(header[dec_card])
                except (TypeError, ValueError):
                    continue
    return None
