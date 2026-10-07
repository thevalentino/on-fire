"""Bundled Magellan/FIRE reference data.

The wavelength reidentification template and the OH airglow line list are
shipped inside the package so that the pipeline can calibrate wavelengths
without any external download (ADR 0001). Both are extracted verbatim from the
canonical PypeIt Magellan/FIRE archive.
"""

from __future__ import annotations

from pathlib import Path

from astropy.io import ascii, fits
from astropy.table import Table

DATA_DIR: Path = Path(__file__).resolve().parent

FIRE_TEMPLATE_PATH: Path = DATA_DIR / "reid_arxiv" / "magellan_fire_echelle.fits"
FIRE_OH_LINES_PATH: Path = DATA_DIR / "arc_lines" / "OH_FIRE_Echelle_lines.dat"


def load_fire_template(path: str | Path | None = None) -> Table:
    """Load the bundled FIRE echelle reidentification archive.

    Args:
        path: Optional override for the archive FITS path.

    Returns:
        An Astropy table with one row per echelle order and columns:

        * ``order``: echelle order number (int).
        * ``wave``: archived wavelength solution on the 2040-pixel grid.
        * ``flux``: archived composite arc spectrum on the same grid.
    """
    with fits.open(Path(path) if path is not None else FIRE_TEMPLATE_PATH, memmap=False) as hdul:
        data = hdul[1].data
        return Table(
            {
                "order": [int(o) for o in data["order"]],
                "wave": [list(w) for w in data["wave"]],
                "flux": [list(f) for f in data["flux"]],
            }
        )


def load_fire_line_list(path: str | Path | None = None) -> Table:
    """Load the bundled FIRE OH airglow line list.

    Args:
        path: Optional override for the line-list path.

    Returns:
        An Astropy table with the rest wavelengths (``wave``, Angstrom), the
        ion label (``ion``) and the relative ``amplitude`` of each line.
    """
    table = ascii.read(
        Path(path) if path is not None else FIRE_OH_LINES_PATH,
        format="fixed_width",
        delimiter="|",
        guess=False,
    )
    return Table(
        {
            "wave": [float(w) for w in table["wave"]],
            "ion": [str(i).strip() for i in table["ion"]],
            "amplitude": [float(a) for a in table["amplitude"]],
        }
    )


__all__ = [
    "DATA_DIR",
    "FIRE_OH_LINES_PATH",
    "FIRE_TEMPLATE_PATH",
    "load_fire_line_list",
    "load_fire_template",
]
