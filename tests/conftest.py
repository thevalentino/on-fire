import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def make_frame(tmp_path):
    """Return a factory that writes a raw FIRE-looking FITS file and returns its path."""

    counter = {"n": 0}

    def _make(obstype="Science", object_name="HIP12345", exptime=900.0,
              acqtime=2458923.123456, airmass=1.234, grism="J", instrume="FIRE",
              extra=None, name=None):
        counter["n"] += 1
        name = name or f"frame{counter['n']:03d}.fits"
        path = tmp_path / name
        hdu = fits.PrimaryHDU(data=np.zeros((10, 10), dtype=np.float32))
        hdu.header["OBSTYPE"] = obstype
        hdu.header["OBJECT"] = object_name
        hdu.header["EXPTIME"] = exptime
        hdu.header["ACQTIME"] = acqtime
        hdu.header["AIRMASS"] = airmass
        hdu.header["GRISM"] = grism
        hdu.header["INSTRUME"] = instrume
        for key, value in (extra or {}).items():
            hdu.header[key] = value
        hdu.writeto(path)
        return path

    return _make
