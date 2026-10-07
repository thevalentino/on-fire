"""Canonical Magellan/FIRE optical definitions.

These constants are extracted from PypeIt's Magellan/FIRE echelle spectrograph
class: the 21 orders (11-31) and their expected fractional spatial positions on
the detector, with index ``i`` corresponding to ``FIRE_ORDER_NUMBERS[i]``. The
positions are for data taken after the 2016 FIRE dewar repair. They are bundled
here so the pipeline can locate and trace orders without any external download
(ADR 0001).
"""

from __future__ import annotations

import numpy as np

FIRE_ORDER_NUMBERS: np.ndarray = np.arange(31, 10, -1, dtype=int)

FIRE_NORDERS: int = int(FIRE_ORDER_NUMBERS.size)

FIRE_ORDER_SPAT_POS: np.ndarray = np.array(
    [
        0.078125,
        0.13769531,
        0.19189453,
        0.24414062,
        0.29296875,
        0.34179688,
        0.38330078,
        0.42724609,
        0.46582031,
        0.50439453,
        0.54199219,
        0.57763672,
        0.61279297,
        0.6484375,
        0.68457031,
        0.71875,
        0.75439453,
        0.79443359,
        0.83789062,
        0.88671875,
        0.94091797,
    ],
    dtype=float,
)
