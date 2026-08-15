"""Standard image preprocessing."""

import numpy as np
from PIL import Image

from core.config.app_config import AppConfig
from core.contracts.image import PreprocessedImage, ValidatedImage
from core.logging import get_logger

logger = get_logger(__name__)


class StandardPreprocessor:
    """Prepares validated images for display and YOLO inference.

    Responsibilities:
        - Preserve display-resolution pixels.
        - Resize inference buffer to configured YOLO input size.

    Dependencies:
        - AppConfig (injected)
    """

    def __init__(self, config: AppConfig) -> None:
        self._inference_size = config.image.yolo_inference_size

    def preprocess(self, image: ValidatedImage) -> PreprocessedImage:
        """Create display and inference pixel arrays.

        Args:
            image: Validated source image.

        Returns:
            PreprocessedImage with separate display and inference buffers.
        """
        display_pixels = image.pixels
        pil_image = Image.fromarray(display_pixels)
        resized = pil_image.resize(
            (self._inference_size, self._inference_size),
            Image.Resampling.LANCZOS,
        )
        inference_pixels = np.asarray(resized, dtype=np.uint8)
        logger.debug(
            "Preprocessed image to inference size %dx%d",
            self._inference_size,
            self._inference_size,
        )
        return PreprocessedImage(
            source=image,
            display_pixels=display_pixels,
            inference_pixels=inference_pixels,
            inference_width=self._inference_size,
            inference_height=self._inference_size,
        )
