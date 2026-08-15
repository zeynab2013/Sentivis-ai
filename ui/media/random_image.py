"""Download a random public-domain / research-dataset image for demos."""

from __future__ import annotations

import random
import socket
import urllib.error
import urllib.request
from pathlib import Path

from core.logging import get_logger
from core.utils.paths import uploads_dir

logger = get_logger(__name__)

# Curated public image endpoints from competition-relevant datasets.
# COCO val2017 CDN, Hugging Face dataset resolves, Open Images samples.
_COCO_IDS = (
    139, 285, 632, 724, 785, 872, 885, 1000, 1204, 1364,
    1440, 1564, 1642, 1725, 1845, 2040, 2141, 2306, 2492, 2618,
    2850, 3072, 3180, 3312, 3460, 3600, 3778, 3920, 4090, 4240,
    4456, 4600, 4780, 4900, 5120, 5320, 5480, 5620, 5800, 6000,
)
_HF_FLICKR = (
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/1000092795.jpg",
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/10002456.jpg",
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/1000268201.jpg",
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/1000344755.jpg",
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images/1000366164.jpg",
)
_OPEN_IMAGES = (
    "https://storage.googleapis.com/openimages/2018_04/test/000026e7ee790996.jpg",
    "https://storage.googleapis.com/openimages/2018_04/test/000062a39995e348.jpg",
    "https://storage.googleapis.com/openimages/2018_04/test/0000c64e1253d68f.jpg",
)
# Visual Genome public image CDN (VG_100K).
_VISUAL_GENOME = tuple(
    f"https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg"
    for image_id in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610)
)
_NSFW_TOKENS = frozenset(
    {
        "nsfw",
        "nude",
        "naked",
        "porn",
        "xxxc",
        "erotic",
        "sex",
        "hentai",
    }
)


def is_online(timeout: float = 1.5) -> bool:
    """Return True when outbound HTTPS is likely available."""
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except OSError:
        try:
            with socket.create_connection(("images.cocodataset.org", 80), timeout=timeout):
                return True
        except OSError:
            return False


def _candidate_urls() -> list[str]:
    urls: list[str] = []
    for image_id in _COCO_IDS:
        urls.append(f"http://images.cocodataset.org/val2017/{image_id:012d}.jpg")
    urls.extend(_HF_FLICKR)
    urls.extend(_OPEN_IMAGES)
    urls.extend(_VISUAL_GENOME)
    random.shuffle(urls)
    return urls


def _is_blocked(url: str) -> bool:
    lower = url.lower()
    return any(token in lower for token in _NSFW_TOKENS)


def _valid_image_bytes(payload: bytes) -> bool:
    if len(payload) < 2048 or len(payload) > 25_000_000:
        return False
    # JPEG / PNG / WEBP magic
    if payload[:3] == b"\xff\xd8\xff":
        return True
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return True
    return False


def fetch_random_public_image(*, max_attempts: int = 8) -> Path:
    """Download a random public dataset image into the uploads directory."""
    if not is_online():
        raise ConnectionError("Offline — random image download is unavailable.")

    dest_dir = uploads_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for url in _candidate_urls()[:max_attempts]:
        if _is_blocked(url):
            continue
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SentivisAI/1.0 (competition demo; research datasets)"},
            )
            with urllib.request.urlopen(request, timeout=12.0) as response:
                payload = response.read()
            if not _valid_image_bytes(payload):
                continue
            suffix = ".jpg"
            if payload[:8] == b"\x89PNG\r\n\x1a\n":
                suffix = ".png"
            out = dest_dir / f"random_{random.randint(100000, 999999)}{suffix}"
            out.write_bytes(payload)
            # Soft decode check via Pillow when available.
            try:
                from PIL import Image

                with Image.open(out) as image:
                    image.verify()
                with Image.open(out) as image:
                    if min(image.size) < 64:
                        out.unlink(missing_ok=True)
                        continue
            except Exception:
                out.unlink(missing_ok=True)
                continue
            logger.info("Random image downloaded from %s → %s", url, out.name)
            return out
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.debug("Random image skip %s: %s", url, exc)
            continue

    raise ConnectionError(f"Could not download a valid public image: {last_error}")
