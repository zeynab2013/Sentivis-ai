"""Unit tests for Streamlit media helpers (UI layer only)."""

from ui.media.random_image import _is_blocked, _valid_image_bytes


def test_nsfw_url_blocked() -> None:
    assert _is_blocked("https://example.com/nsfw_photo.jpg")
    assert not _is_blocked("https://images.cocodataset.org/val2017/000000000139.jpg")


def test_valid_image_bytes_rejects_tiny_payload() -> None:
    assert not _valid_image_bytes(b"not-an-image")
    # Minimal JPEG header alone is still too small for our size gate.
    jpeg_tiny = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    assert not _valid_image_bytes(jpeg_tiny)
