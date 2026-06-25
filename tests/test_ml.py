"""comicml: split safety, pixel-error metric, and the flip-TTA corner
reordering that must stay in lockstep between training augmentation and
inference TTA.

The canonical horizontal-flip mapping for [TL, TR, BR, BL] corners is:
mirror x, then swap TL↔TR and BL↔BR. Both train.py's augmentation and
inference.py's TTA must implement exactly this; if either drifts, accuracy
silently degrades. Both are tested here against the same reference mapping.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from comicml.train import (  # noqa: E402
    _split_entries, _resolve_train_dirs, _corner_px_error, PageCornerDataset,
)
from comicml.inference import predict_corners  # noqa: E402


def flip_map(norm):
    """Reference horizontal-flip mapping on normalized [TL,TR,BR,BL] corners."""
    flipped = norm.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return flipped[[1, 0, 3, 2]]


# ---------------------------------------------------------------------------
# Split + metric
# ---------------------------------------------------------------------------

def _entry(scan_dir, page=0):
    return {"scan_dir": f"/scans/{scan_dir}", "page_index": page,
            "filepath": f"/scans/{scan_dir}/Scan {page}.jpeg",
            "gt_rotate180": False, "gt_corners": [[0, 0]] * 4}


class TestSplitEntries:
    def test_split_by_dir(self):
        entries = [_entry("A"), _entry("B"), _entry("C"), _entry("A", 1)]
        train, holdout = _split_entries(entries, ["A"], ["B"])
        assert len(train) == 2 and len(holdout) == 1
        # "C" belongs to neither and is excluded
        assert all(e["scan_dir"].endswith("A") for e in train)

    def test_overlap_rejected(self):
        with pytest.raises(ValueError, match="both train and holdout"):
            _split_entries([], ["A", "B"], ["B", "C"])


class TestResolveTrainDirs:
    def test_default_is_all_dirs_minus_holdout(self):
        # The footgun guard: every collected dir except the holdout, so newly
        # collected comics are never silently dropped from training.
        entries = [_entry("A"), _entry("B"), _entry("C"), _entry("NEW")]
        assert _resolve_train_dirs(entries, ["B"]) == ["A", "C", "NEW"]

    def test_explicit_overrides_default(self):
        entries = [_entry("A"), _entry("B"), _entry("C")]
        assert _resolve_train_dirs(entries, ["B"], ["A"]) == ["A"]


class TestCornerPxError:
    def test_known_offset(self):
        # Prediction off by exactly (0.01, 0) normalized on a 1000px-wide image
        target = torch.zeros(1, 8)
        pred = target.clone()
        pred[0, 0::2] += 0.01  # shift all x coords
        w = torch.tensor([1000.0])
        h = torch.tensor([500.0])
        mean_px, per_image = _corner_px_error(pred, target, w, h)
        assert mean_px == pytest.approx(10.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Flip handling: dataset augmentation vs reference mapping
# ---------------------------------------------------------------------------

class TestDatasetFlip:
    CORNERS = [[10, 20], [90, 22], [88, 180], [12, 178]]

    def _dataset_entry(self, tmp_path, make_page_image):
        path = make_page_image("Scan.jpeg", size=(100, 200))
        return {"filepath": str(path), "gt_rotate180": False,
                "gt_corners": self.CORNERS, "scan_dir": str(tmp_path),
                "page_index": 0}

    def test_no_augment_normalizes(self, tmp_path, make_page_image):
        entry = self._dataset_entry(tmp_path, make_page_image)
        ds = PageCornerDataset([entry], augment=False, input_size=64)
        _, target, meta = ds[0]
        norm = np.array(self.CORNERS, dtype=np.float32) / [100, 200]
        assert np.allclose(target.numpy(), norm.flatten(), atol=1e-6)
        assert meta["orig_w"] == 100 and meta["orig_h"] == 200

    def test_flip_matches_reference(self, tmp_path, make_page_image, monkeypatch):
        entry = self._dataset_entry(tmp_path, make_page_image)
        ds = PageCornerDataset([entry], augment=True, input_size=64)
        # Force the flip branch and suppress every other augmentation
        rolls = iter([0.0] + [0.99] * 20)
        monkeypatch.setattr("comicml.train.random.random", lambda: next(rolls))
        _, target, _ = ds[0]
        norm = np.array(self.CORNERS, dtype=np.float32) / [100, 200]
        assert np.allclose(target.numpy(), flip_map(norm).flatten(), atol=1e-6)

    def test_flip_is_involution(self):
        rng = np.random.default_rng(7)
        norm = rng.random((4, 2)).astype(np.float32)
        assert np.allclose(flip_map(flip_map(norm)), norm)


# ---------------------------------------------------------------------------
# Flip handling: inference TTA vs reference mapping
# ---------------------------------------------------------------------------

class _ConstModel(torch.nn.Module):
    """Always predicts the same normalized corners, regardless of input."""

    def __init__(self, norm_corners):
        super().__init__()
        self.out = torch.tensor(norm_corners, dtype=torch.float32).reshape(1, 8)

    def forward(self, x):
        # Clone: a real model returns a fresh tensor each forward, and
        # _predict_single scales its output to pixel space in place.
        return self.out.expand(x.shape[0], 8).clone()


class TestInferenceTTA:
    W, H = 400, 600

    def _img(self):
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)

    def test_symmetric_corners_unchanged_by_tta(self):
        # Horizontally symmetric prediction: TTA average must reproduce it
        # exactly — only true if the un-flip corner reordering is correct.
        norm = np.array([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]])
        model = _ConstModel(norm)
        corners = predict_corners(model, "cpu", self._img(), input_size=64)
        expected = norm * [self.W, self.H]
        assert np.allclose(corners, expected, atol=1e-3)

    def test_asymmetric_corners_average_with_reference_flip(self):
        norm = np.array([[0.10, 0.18], [0.85, 0.22], [0.88, 0.79], [0.13, 0.81]])
        model = _ConstModel(norm)
        corners = predict_corners(model, "cpu", self._img(), input_size=64)
        scale = np.array([self.W, self.H])
        expected = (norm * scale + flip_map(norm) * scale) / 2.0
        assert np.allclose(corners, expected, atol=1e-3)

    def test_tta_disabled_returns_raw(self):
        norm = np.array([[0.10, 0.18], [0.85, 0.22], [0.88, 0.79], [0.13, 0.81]])
        model = _ConstModel(norm)
        corners = predict_corners(model, "cpu", self._img(), input_size=64,
                                  tta=False)
        assert np.allclose(corners, norm * [self.W, self.H], atol=1e-3)
