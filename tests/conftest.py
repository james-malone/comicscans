"""Shared fixtures. Adds the project root to sys.path so tests import the
top-level modules (comicscans, comicpackage, comiceval) and the comicml
package without an install step."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402


@pytest.fixture
def make_page_image(tmp_path):
    """Factory: write a small solid-color page image and return its path."""

    def _make(name, size=(300, 450), color=(240, 230, 210), directory=None):
        d = directory or tmp_path
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        Image.new('RGB', size, color).save(path, dpi=(300, 300))
        return path

    return _make


@pytest.fixture
def pages_dir(tmp_path, make_page_image):
    """A directory with 3 normal processed pages: Scan 0/1/2.jpg."""
    d = tmp_path / "pages"
    for i in range(3):
        # Vary the color so pages aren't flagged as duplicates of each other
        make_page_image(f"Scan {i}.jpg", color=(240 - 40 * i, 100 + 50 * i, 60),
                        directory=d)
    return d
