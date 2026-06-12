"""Scan filename parsing and page-file discovery."""

import pytest

from comicscans import parse_scan_filename, load_scans
from comicpackage import find_page_files


class TestParseScanFilename:
    @pytest.mark.parametrize("name,expected", [
        ("Scan.jpeg", 0),
        ("Scan 1.jpeg", 1),
        ("Scan 35.jpeg", 35),
        ("Scan 007.jpg", 7),
        ("IMG_0042.jpg", None),
        ("Scanner.jpg", None),         # 'Scan' prefix but not a scan file
        ("Scan extra notes.jpg", None),  # non-numeric suffix
    ])
    def test_parse(self, name, expected):
        assert parse_scan_filename(name) == expected


class TestLoadScans:
    def test_sorted_numerically(self, tmp_path, make_page_image):
        # Created out of order; index 10 sorts after 2 (not lexicographic)
        for name in ["Scan 10.jpeg", "Scan.jpeg", "Scan 2.jpeg", "Scan 1.jpeg"]:
            make_page_image(name)
        scans = load_scans(str(tmp_path))
        assert [idx for idx, _ in scans] == [0, 1, 2, 10]
        assert scans[0][1].name == "Scan.jpeg"

    def test_ignores_non_scan_files(self, tmp_path, make_page_image):
        make_page_image("Scan.jpeg")
        make_page_image("cover-notes.jpg")
        (tmp_path / "notes.txt").write_text("not an image")
        scans = load_scans(str(tmp_path))
        assert len(scans) == 1

    def test_empty_dir_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_scans(str(tmp_path))


class TestFindPageFiles:
    def test_numeric_sort(self, tmp_path, make_page_image):
        d = tmp_path / "out"
        for i in [10, 0, 2, 1]:
            make_page_image(f"Scan {i}.jpg", directory=d)
        files = find_page_files(d)
        assert [f.name for f in files] == \
            ["Scan 0.jpg", "Scan 1.jpg", "Scan 2.jpg", "Scan 10.jpg"]

    def test_no_space_variant(self, tmp_path, make_page_image):
        d = tmp_path / "out"
        for i in [1, 0]:
            make_page_image(f"Scan{i}.webp", directory=d)
        files = find_page_files(d)
        assert [f.name for f in files] == ["Scan0.webp", "Scan1.webp"]

    def test_bare_scan_is_page_zero(self, tmp_path, make_page_image):
        # 'Scan.jpg' (the cover in the scanner's naming scheme) must not be
        # dropped when numbered pages are present — this was a real bug.
        d = tmp_path / "out"
        make_page_image("Scan 1.jpg", directory=d)
        make_page_image("Scan.jpg", directory=d)
        files = find_page_files(d)
        assert [f.name for f in files] == ["Scan.jpg", "Scan 1.jpg"]

    def test_junk_files_ignored(self, tmp_path, make_page_image):
        # Non-numeric suffixes used to raise ValueError inside the sort key
        d = tmp_path / "out"
        make_page_image("Scan 0.jpg", directory=d)
        make_page_image("Scan extra notes.jpg", directory=d)
        (d / "ComicInfo.xml").write_text("<ComicInfo/>")
        files = find_page_files(d)
        assert [f.name for f in files] == ["Scan 0.jpg"]

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir()
        assert find_page_files(d) == []
