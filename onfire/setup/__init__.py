"""Setup: ingest raw frames and generate the fixed-width Frame Table."""

from onfire.setup.frames import (
    FRAME_TYPES,
    METADATA_CARDS,
    Frame,
    classify_frame,
    read_frame,
    scan_raw_dir,
)
from onfire.setup.generate import generate_frame_table
from onfire.setup.table import COLUMNS, Column, FrameTable
from onfire.setup.validate import REQUIRED_CALIBRATIONS, validate_calibrations

__all__ = [
    "COLUMNS",
    "FRAME_TYPES",
    "METADATA_CARDS",
    "REQUIRED_CALIBRATIONS",
    "Column",
    "Frame",
    "FrameTable",
    "classify_frame",
    "generate_frame_table",
    "read_frame",
    "scan_raw_dir",
    "validate_calibrations",
]
