"""Session persistence: save → load must round-trip detections, overrides,
and ignored flags through .comicscans_session.json."""

import pytest

torch = pytest.importorskip("torch")  # webapp server imports comicml → torch

from webapp.scan import server  # noqa: E402


def _fresh_session(input_dir):
    return {
        "input_dir": str(input_dir),
        "detection": {},
        "overrides": {},
        "ignored": {},
    }


def test_roundtrip(tmp_path):
    session = _fresh_session(tmp_path)
    session["detection"][0] = {
        "corners": [[10, 20], [400, 22], [398, 600], [12, 598]],
        "rotation": 0.4,
        "rotate180": False,
        "bleed_method": "cnn+snap",
    }
    session["overrides"][1] = {
        "corners": [[5, 5], [395, 5], [395, 595], [5, 595]],
        "rotation": -0.2,
        "rotate180": True,
    }
    session["ignored"][2] = True

    server._save_session_file(session)
    assert (tmp_path / server.SAVE_FILENAME).exists()

    loaded = _fresh_session(tmp_path)
    server._load_session_file(loaded)

    assert loaded["detection"][0]["corners"] == session["detection"][0]["corners"]
    assert loaded["detection"][0]["rotation"] == 0.4
    assert loaded["detection"][0]["bleed_method"] == "cnn+snap"
    assert loaded["overrides"][1]["rotate180"] is True
    assert loaded["ignored"] == {2: True}


def test_load_missing_file_is_noop(tmp_path):
    session = _fresh_session(tmp_path)
    server._load_session_file(session)
    assert session["detection"] == {} and session["overrides"] == {}


def test_load_corrupt_file_is_noop(tmp_path):
    (tmp_path / server.SAVE_FILENAME).write_text("{not json")
    session = _fresh_session(tmp_path)
    server._load_session_file(session)
    assert session["detection"] == {}


def test_clear_session_file(tmp_path):
    session = _fresh_session(tmp_path)
    session["detection"][0] = {"corners": [[0, 0]] * 4, "rotation": 0,
                               "rotate180": False, "bleed_method": None}
    server._save_session_file(session)
    assert server._clear_session_file(session) is True
    assert not (tmp_path / server.SAVE_FILENAME).exists()
    assert server._clear_session_file(session) is False
