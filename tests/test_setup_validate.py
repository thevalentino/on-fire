import pytest

from onfire.setup import (
    FrameTable,
    generate_frame_table,
    scan_raw_dir,
    validate_calibrations,
)


def _table(make_frame, tmp_path, obstypes):
    for i, obstype in enumerate(obstypes):
        make_frame(obstype=obstype, name=f"f{i}.fits")
    return FrameTable.from_frames(scan_raw_dir(tmp_path))


def test_validate_calibrations_reports_missing_flats_and_arcs(make_frame, tmp_path):
    table = _table(make_frame, tmp_path, ["Science", "Telluric"])

    messages = validate_calibrations(table)

    assert any("flat" in m.lower() for m in messages)
    assert any("arc" in m.lower() for m in messages)


def test_validate_calibrations_is_silent_when_present(make_frame, tmp_path):
    table = _table(make_frame, tmp_path, ["PixFlat", "Arc", "Science"])

    assert validate_calibrations(table) == []


def test_validate_calibrations_reports_one_missing_type(make_frame, tmp_path):
    table = _table(make_frame, tmp_path, ["PixFlat", "Science"])

    messages = validate_calibrations(table)

    assert len(messages) == 1
    assert "arc" in messages[0].lower()


def test_generate_frame_table_writes_file_and_warns(make_frame, tmp_path):
    make_frame(obstype="Science", name="a.fits")
    out = tmp_path / "frames.ascii"

    with pytest.warns(UserWarning):
        table = generate_frame_table(tmp_path, output=out)

    assert out.exists()
    assert len(table.frames) == 1
    assert FrameTable.read(out).to_rows() == table.to_rows()


def test_generate_frame_table_refuses_to_clobber_without_overwrite(make_frame, tmp_path):
    make_frame(obstype="PixFlat", name="a.fits")
    make_frame(obstype="Arc", name="b.fits")
    out = tmp_path / "frames.ascii"
    out.write_text("existing")

    with pytest.raises(FileExistsError):
        generate_frame_table(tmp_path, output=out)

    assert out.read_text() == "existing"
