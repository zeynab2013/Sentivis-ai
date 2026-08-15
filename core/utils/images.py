"""Image utility helpers without ML dependencies."""

from pathlib import Path

from PIL import Image


def load_image_rgb(path: Path) -> Image.Image:
    """Load an image from disk in RGB mode.

    Args:
        path: Path to image file.

    Returns:
        PIL Image in RGB mode.
    """
    with Image.open(path) as image:
        return image.convert("RGB")
