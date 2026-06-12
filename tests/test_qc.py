"""comicpackage QC checks against tiny synthetic page sets."""

import shutil

from PIL import Image

from comicpackage import run_qc


def test_qc_passes_on_clean_pages(pages_dir, capsys):
    # 3 pages trip the "unusual page count" warning, so build 24 distinct ones
    for i in range(3, 24):
        Image.new('RGB', (300, 450), (10 * i, 255 - 10 * i, 60)).save(
            pages_dir / f"Scan {i}.jpg", dpi=(300, 300))
    # Solid colors look "blank" (zero stddev) — add texture
    import numpy as np
    rng = np.random.default_rng(0)
    for f in pages_dir.glob("Scan *.jpg"):
        arr = rng.integers(0, 255, (450, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(f, dpi=(300, 300))
    assert run_qc(pages_dir) is True


def test_qc_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert run_qc(d) is False


def test_qc_flags_blank_page(pages_dir, capsys):
    # Solid-color fixture pages have ~zero stddev → blank
    assert run_qc(pages_dir) is False
    out = capsys.readouterr().out
    assert "appears blank" in out


def test_qc_flags_duplicates(pages_dir, capsys):
    shutil.copy(pages_dir / "Scan 0.jpg", pages_dir / "Scan 3.jpg")
    run_qc(pages_dir)
    out = capsys.readouterr().out
    assert "Possible duplicate" in out


def test_qc_survives_corrupt_file(pages_dir, capsys):
    (pages_dir / "Scan 3.jpg").write_text("this is not a jpeg")
    # Must not raise; corrupt page reported as unreadable + integrity failure
    assert run_qc(pages_dir) is False
    out = capsys.readouterr().out
    assert "unreadable" in out
    assert "corrupt" in out


def test_qc_all_corrupt(tmp_path, capsys):
    d = tmp_path / "allbad"
    d.mkdir()
    (d / "Scan 0.jpg").write_text("garbage")
    assert run_qc(d) is False
    out = capsys.readouterr().out
    assert "No readable pages" in out
