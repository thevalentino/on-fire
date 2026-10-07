from pathlib import Path

import pytest

from onfire.setup import FrameTable, scan_raw_dir


def _frames(make_frame, tmp_path):
    make_frame(obstype="PixFlat", object_name="Dome Flat", exptime=5.0, name="a.fits")
    make_frame(obstype="Science", object_name="HIP12345", exptime=900.0, name="b.fits")
    make_frame(obstype="Telluric", object_name="HD224423", exptime=30.0, name="c.fits")
    return scan_raw_dir(tmp_path)


def test_frame_table_writes_expected_columns(make_frame, tmp_path):
    table = FrameTable.from_frames(_frames(make_frame, tmp_path))

    text = table.to_string()
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]

    assert "filename" in text
    assert "frame_type" in text
    assert "exptime" in text
    assert len({len(ln) for ln in data_lines}) == 1
    assert data_lines[0].split()[1] == "a.fits"


def test_frame_table_roundtrips_through_disk(make_frame, tmp_path):
    table = FrameTable.from_frames(_frames(make_frame, tmp_path))
    out = tmp_path / "frames.ascii"

    table.write(out)
    reloaded = FrameTable.read(out)

    assert out.exists()
    assert reloaded.to_rows() == table.to_rows()
    assert [f.frame_type for f in reloaded.frames] == ["pixelflat", "science", "standard"]


def test_frame_table_is_editable_as_fixed_width_text(make_frame, tmp_path):
    table = FrameTable.from_frames(_frames(make_frame, tmp_path))

    edited = table.to_string().replace("pixelflat", "arc      ", 1)
    reloaded = FrameTable.from_string(edited)

    assert reloaded.frames[0].frame_type == "arc"


def test_frame_table_from_string_preserves_file_order(make_frame, tmp_path):
    table = FrameTable.from_frames(_frames(make_frame, tmp_path))

    reloaded = FrameTable.from_string(table.to_string())

    assert [Path(f.filename).name for f in reloaded.frames] == [
        "a.fits",
        "b.fits",
        "c.fits",
    ]
    assert [row["idx"] for row in reloaded.to_rows()] == ["1", "2", "3"]


def test_frame_table_preserves_long_filenames(make_frame, tmp_path):
    name = "fire_20240215_HIP12345_J_0900s.fits"
    make_frame(obstype="Science", name=name)

    table = FrameTable.from_frames(scan_raw_dir(tmp_path))
    reloaded = FrameTable.from_string(table.to_string())

    assert reloaded.frames[0].filename == name


def test_frame_table_rejects_values_wider_than_column(make_frame, tmp_path):
    make_frame(obstype="Science", object_name="x" * 200, name="a.fits")

    table = FrameTable.from_frames(scan_raw_dir(tmp_path))

    with pytest.raises(ValueError, match="does not fit column"):
        table.to_string()
