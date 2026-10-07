"""Validation of required calibration coverage in a Frame Table."""

from __future__ import annotations

from onfire.setup.table import FrameTable

# Calibration frame roles that must be present for a reduction to proceed.
REQUIRED_CALIBRATIONS: tuple[str, ...] = ("pixelflat", "arc")

_LABELS: dict[str, str] = {
    "pixelflat": "flat-field (pixelflat) frames",
    "arc": "arc frames",
}


def validate_calibrations(table: FrameTable) -> list[str]:
    """Return a human-readable message for each missing required calibration.

    An empty list means the Frame Table provides all required calibrations.
    """
    present = {frame.frame_type for frame in table.frames}
    return [
        f"missing required {_LABELS.get(required, required)}"
        for required in REQUIRED_CALIBRATIONS
        if required not in present
    ]
