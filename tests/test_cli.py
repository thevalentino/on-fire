from onfire.cli import main
from onfire.setup import FrameTable


def test_cli_setup_writes_frame_table(make_frame, tmp_path):
    make_frame(obstype="PixFlat", name="a.fits")
    make_frame(obstype="Arc", name="b.fits")
    out = tmp_path / "frames.ascii"

    rc = main(["setup", str(tmp_path), "--output", str(out)])

    assert rc == 0
    assert out.exists()
    assert len(FrameTable.read(out).frames) == 2


def test_cli_setup_errors_on_missing_directory(tmp_path, capsys):
    rc = main(["setup", str(tmp_path / "does_not_exist")])

    assert rc == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_cli_setup_errors_when_output_exists(make_frame, tmp_path, capsys):
    make_frame(obstype="PixFlat", name="a.fits")
    make_frame(obstype="Arc", name="b.fits")
    out = tmp_path / "frames.ascii"
    out.write_text("existing")

    rc = main(["setup", str(tmp_path), "--output", str(out)])

    assert rc == 2
    assert "exists" in capsys.readouterr().err.lower()
    assert out.read_text() == "existing"
