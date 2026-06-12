"""Geometry: bounds→corners conversion and perspective crop.

The webapp server and comiceval both convert detect_page_bounds() output to
original-image corners — they must agree, since the webapp's numbers become
the ground truth that comiceval evaluates against.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # webapp server imports comicml → torch

from webapp.scan import server  # noqa: E402
import comiceval  # noqa: E402


BOUNDS_STRAIGHT = {"angle": 0.0, "top": 100, "bottom": 6000,
                   "left": 200, "right": 4100, "spine_col": None}
BOUNDS_SKEWED = {"angle": 1.3, "top": 100, "bottom": 6000,
                 "left": 200, "right": 4100, "spine_col": None}


class TestBoundsToCorners:
    def test_zero_angle_passthrough(self):
        corners = server._bounds_to_original_corners(BOUNDS_STRAIGHT, 4400, 6200)
        assert corners == [[200, 100], [4100, 100], [4100, 6000], [200, 6000]]

    def test_skew_maps_back_inside_image(self):
        w, h = 4400, 6200
        corners = server._bounds_to_original_corners(BOUNDS_SKEWED, w, h)
        # A small rotation keeps an interior rectangle inside the image
        for x, y in corners:
            assert -50 <= x <= w + 50
            assert -50 <= y <= h + 50
        # Corner order TL, TR, BR, BL preserved
        (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = corners
        assert tlx < trx and blx < brx
        assert tly < bly and try_ < bry

    @pytest.mark.parametrize("bounds", [BOUNDS_STRAIGHT, BOUNDS_SKEWED])
    def test_server_and_comiceval_agree(self, monkeypatch, bounds):
        """comiceval.run_detection re-implements the same conversion inline —
        it must produce identical corners for identical bounds."""
        w, h = 4400, 6200
        monkeypatch.setattr(comiceval, "detect_page_bounds",
                            lambda image, dpi, params=None: dict(bounds))
        entry = {"filepath": "unused", "dpi": 300, "gt_rotate180": False}
        fake_img = np.zeros((h, w, 3), dtype=np.uint8)
        det = comiceval.run_detection(entry, params={}, preloaded_image=fake_img)
        expected = server._bounds_to_original_corners(bounds, w, h)
        assert np.allclose(det["corners"], expected, atol=0.11)


class TestPerspectiveCrop:
    def test_axis_aligned_crop(self):
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        img[50:350, 40:260] = (10, 200, 30)  # BGR block
        corners = [[40, 50], [260, 50], [260, 350], [40, 350]]
        out = server.perspective_crop(img, corners)
        assert out.shape[0] == pytest.approx(300, abs=2)
        assert out.shape[1] == pytest.approx(220, abs=2)
        # Interior should be uniformly the block color
        interior = out[10:-10, 10:-10]
        assert (interior == (10, 200, 30)).all()
