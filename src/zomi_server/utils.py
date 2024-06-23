from logging import getLogger

import cv2
import numpy as np

from .Log import SERVER_LOGGER_NAME

logger = getLogger(SERVER_LOGGER_NAME)


def resize_cv2_image(img: np.ndarray, resize_w: int):
    """Resize a CV2 (numpy.ndarray) image using ``resize_w``. Width is used and height will be scaled accordingly."""
    lp = "resize:img:"

    if img is not None:
        h, w = img.shape[:2]
        aspect_ratio: float = resize_w / w
        dim: tuple = (resize_w, int(h * aspect_ratio))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        logger.debug(
            f"{lp} success using resize_width={resize_w} - original dimensions: {w}*{h}"
            f" - resized dimensions: {dim[1]}*{dim[0]}",
        )
    else:
        logger.debug(f"{lp} 'resize' called but no image supplied!")
    return img
