"""Unit tests for Streamlit branding paths."""

from streamlit_app.branding import favicon_path, logo_path


def test_logo_path_exists() -> None:
    path = logo_path()
    assert path.is_file()


def test_favicon_path_exists() -> None:
    path = favicon_path()
    assert path.is_file()
