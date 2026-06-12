"""Ground-truth lint: geometric sanity checks in comiceval."""

from comiceval import quad_convex, quad_dimensions, lint_entry, lint_ground_truth


GOOD = [[200, 150], [4100, 160], [4090, 6000], [210, 5990]]  # TL, TR, BR, BL


def _entry(corners, scan_dir="/scans/A", page=0, w=4960, h=7016, corrected=True):
    return {"scan_dir": scan_dir, "page_index": page,
            "filepath": f"{scan_dir}/Scan {page}.jpeg",
            "image_width": w, "image_height": h,
            "gt_corners": corners, "has_correction": corrected,
            "gt_rotate180": False}


class TestQuadConvex:
    def test_good_quad(self):
        assert quad_convex(GOOD) is True

    def test_swapped_corners(self):
        # TR and BR swapped → self-intersecting bow-tie
        bad = [GOOD[0], GOOD[2], GOOD[1], GOOD[3]]
        assert quad_convex(bad) is False

    def test_counterclockwise_order(self):
        # Reversed winding (TL, BL, BR, TR) is mis-labeled corners
        assert quad_convex([GOOD[0], GOOD[3], GOOD[2], GOOD[1]]) is False

    def test_concave_quad(self):
        # BR pulled deep inside the quad
        bad = [GOOD[0], GOOD[1], [1000, 2000], GOOD[3]]
        assert quad_convex(bad) is False


class TestQuadDimensions:
    def test_axis_aligned(self):
        w, h = quad_dimensions([[0, 0], [400, 0], [400, 600], [0, 600]])
        assert w == 400 and h == 600


class TestLintEntry:
    def test_clean_entry(self):
        assert lint_entry(_entry(GOOD)) == []

    def test_corner_outside_image(self):
        bad = [GOOD[0], [4991, 160], GOOD[2], GOOD[3]]
        issues = lint_entry(_entry(bad, w=4960))
        assert any("outside image bounds" in i for i in issues)

    def test_non_convex_flagged(self):
        bad = [GOOD[0], GOOD[2], GOOD[1], GOOD[3]]
        issues = lint_entry(_entry(bad))
        assert any("convex" in i for i in issues)

    def test_aspect_ratio_flagged(self):
        # Nearly square crop — not a comic page
        square = [[0, 0], [4000, 0], [4000, 4100], [0, 4100]]
        issues = lint_entry(_entry(square))
        assert any("aspect ratio" in i for i in issues)

    def test_degenerate_flagged(self):
        tiny = [[0, 0], [30, 0], [30, 40], [0, 40]]
        issues = lint_entry(_entry(tiny))
        assert any("degenerate" in i for i in issues)


class TestPerDirOutliers:
    def test_size_outlier_flagged(self):
        # 9 consistent pages + 1 whose width is ~300px off
        entries = []
        for p in range(9):
            jitter = (p % 3) - 1  # ±1px noise so MAD isn't zero
            c = [[200, 150], [4100 + jitter, 150], [4100 + jitter, 6000], [200, 6000]]
            entries.append(_entry(c, page=p))
        outlier = [[200, 150], [4400, 150], [4400, 6000], [200, 6000]]
        entries.append(_entry(outlier, page=9))
        report = lint_ground_truth(entries)
        flagged_pages = {e["page_index"] for e, issue in report if "deviates" in issue}
        assert flagged_pages == {9}

    def test_small_dirs_skipped(self):
        # Only 2 pages in the dir → no outlier stats, no false flags
        entries = [_entry(GOOD, page=0),
                   _entry([[200, 150], [4400, 150], [4400, 6000], [200, 6000]], page=1)]
        report = lint_ground_truth(entries, min_dir_pages=5)
        assert report == []

    def test_clean_set_no_issues(self):
        entries = [_entry(GOOD, page=p) for p in range(10)]
        assert lint_ground_truth(entries) == []
