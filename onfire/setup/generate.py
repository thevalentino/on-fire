"""Orchestration for ``onfire setup``: scan, validate, and write the Frame Table."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from onfire.setup.frames import scan_raw_dir
from onfire.setup.table import FrameTable
from onfire.setup.validate import validate_calibrations


def generate_frame_table(
    raw_dir: str | Path,
    output: Optional[str | Path] = "frames.ascii",
    overwrite: bool = False,
) -> FrameTable:
    """Scan ``raw_dir``, classify frames, validate calibrations, and write a table.

    Args:
        raw_dir: Directory containing raw Magellan/FIRE ``*.fits`` exposures.
        output: Destination Frame Table path. If ``None``, nothing is written.
        overwrite: Allow replacing an existing ``output`` file.

    Returns:
        The generated :class:`~onfire.setup.table.FrameTable`.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        FileExistsError: If ``output`` exists and ``overwrite`` is ``False``.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")

    table = FrameTable.from_frames(scan_raw_dir(raw_dir), source=str(raw_dir))

    for message in validate_calibrations(table):
        warnings.warn(message, UserWarning, stacklevel=2)

    if output is not None:
        output = Path(output)
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"{output} already exists; pass overwrite=True to replace it"
            )
        table.write(output)

    return table
