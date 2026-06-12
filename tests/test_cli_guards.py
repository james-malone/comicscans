"""CLI argument validation in comicscans.py."""

from argparse import Namespace

import pytest

from comicscans import parse_rotate_pages


def _args(**kw):
    base = dict(rotate=None, rotate_range=None, rotate_even=False, rotate_odd=False)
    base.update(kw)
    return Namespace(**base)


class TestParseRotatePages:
    def test_single_and_list(self):
        assert parse_rotate_pages(_args(rotate="3"), 10) == {3}
        assert parse_rotate_pages(_args(rotate="1, 4,6"), 10) == {1, 4, 6}

    def test_range(self):
        assert parse_rotate_pages(_args(rotate="2-5"), 10) == {2, 3, 4, 5}
        assert parse_rotate_pages(_args(rotate_range="0-2"), 10) == {0, 1, 2}

    def test_mixed(self):
        assert parse_rotate_pages(_args(rotate="0,2-4,7"), 10) == {0, 2, 3, 4, 7}

    def test_even_odd(self):
        assert parse_rotate_pages(_args(rotate_even=True), 5) == {0, 2, 4}
        assert parse_rotate_pages(_args(rotate_odd=True), 5) == {1, 3}

    @pytest.mark.parametrize("bad", ["foo", "2-foo", "1,bar", "x-3"])
    def test_invalid_rotate_exits(self, bad):
        with pytest.raises(SystemExit):
            parse_rotate_pages(_args(rotate=bad), 10)

    def test_invalid_rotate_range_exits(self):
        with pytest.raises(SystemExit):
            parse_rotate_pages(_args(rotate_range="2-foo"), 10)
