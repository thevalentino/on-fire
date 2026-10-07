import warnings

import pytest

from onfire.setup import read_frame, scan_raw_dir


def test_read_frame_ingests_header_metadata(make_frame):
    path = make_frame(
        obstype="Science",
        object_name="HIP12345",
        exptime=900.0,
        acqtime=2458923.123456,
        airmass=1.234,
        grism="J",
        instrume="FIRE",
    )

    frame = read_frame(path)

    assert frame.filename == "frame001.fits"
    assert frame.obstype == "Science"
    assert frame.object_name == "HIP12345"
    assert frame.exptime == pytest.approx(900.0)
    assert frame.acqtime == pytest.approx(2458923.123456)
    assert frame.airmass == pytest.approx(1.234)
    assert frame.grism == "J"
    assert frame.instrume == "FIRE"


@pytest.mark.parametrize(
    "obstype, expected",
    [
        ("Science", "science"),
        ("Telluric", "standard"),
        ("PixFlat", "pixelflat"),
        ("Arc", "arc"),
    ],
)
def test_read_frame_classifies_frame_type(make_frame, obstype, expected):
    path = make_frame(obstype=obstype, object_name="some target")

    assert read_frame(path).frame_type == expected


def test_read_frame_tolerates_missing_metadata(make_frame):
    path = make_frame(extra=None, name="missing.fits")
    # Remove the metadata cards to simulate a sparse header.
    from astropy.io import fits

    with fits.open(path, mode="update") as hdul:
        for card in ("OBSTYPE", "OBJECT", "EXPTIME", "ACQTIME", "AIRMASS", "GRISM"):
            if card in hdul[0].header:
                del hdul[0].header[card]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = read_frame(path)

    assert frame.obstype == ""
    assert frame.object_name == ""
    assert frame.exptime is None
    assert frame.frame_type == "unknown"
    assert any("missing" in str(w.message).lower() for w in caught)


def test_scan_raw_dir_reads_fits_files_in_order(make_frame, tmp_path):
    make_frame(obstype="Science", name="b.fits")
    make_frame(obstype="PixFlat", name="a.fits")
    make_frame(obstype="Telluric", name="c.fits")

    frames = scan_raw_dir(tmp_path)

    assert [f.filename for f in frames] == ["a.fits", "b.fits", "c.fits"]
    assert [f.frame_type for f in frames] == ["pixelflat", "science", "standard"]


def test_scan_raw_dir_ignores_non_fits_files(make_frame, tmp_path):
    make_frame(obstype="Science", name="a.fits")
    (tmp_path / "notes.txt").write_text("not a frame")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.fits").write_bytes(b"")

    frames = scan_raw_dir(tmp_path)

    assert [f.filename for f in frames] == ["a.fits"]


def test_scan_raw_dir_skips_unreadable_fits_with_warning(make_frame, tmp_path):
    make_frame(obstype="Science", name="good.fits")
    (tmp_path / "corrupt.fits").write_bytes(b"not a real fits file")

    with pytest.warns(UserWarning, match="corrupt.fits"):
        frames = scan_raw_dir(tmp_path)

    assert [f.filename for f in frames] == ["good.fits"]
