"""Unit tests for Streamlit JSON preferences."""

from streamlit_app import preferences as prefs


def test_preferences_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prefs, "_PREFS_PATH", tmp_path / "prefs.json")
    prefs.save_language("es")
    prefs.save_competition_mode(True)
    prefs.save_enable_sam2(False)
    assert prefs.load_language() == "es"
    assert prefs.load_competition_mode() is True
    assert prefs.load_enable_sam2() is False
