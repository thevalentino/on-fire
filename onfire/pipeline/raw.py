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

ROLE_PRODUCTS: dict[str, str] = {
    "pixelflat": "raw_flats",
    "arc": "raw_arcs",
    "science": "raw_science",
    "standard": "raw_standard",
}


def read_fits_data(path: str | Path) -> np.ndarray:
    """Read the primary image data of a raw FITS exposure as float."""
    with fits.open(Path(path), memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"{path}: primary HDU contains no image data")
    return np.asarray(data, dtype=float)


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
        stack = np.stack([read_fits_data(frame.path) for frame in frames])
        key = ROLE_PRODUCTS.get(role, f"raw_{role}")
        state.products[key] = stack
        loaded[key] = stack
    return loaded
