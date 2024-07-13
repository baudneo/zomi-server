import logging
from typing import List, Optional, Tuple

import numpy as np

from ....Log import SERVER_LOGGER_NAME
from ..onnx_runtime import xywh2xyxy

logger = logging.getLogger(SERVER_LOGGER_NAME)
LP: str = "YOLOv8:"


def process_output(output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the output from the YOLOv8 pretrained models (1, <num_classes + 4>, 8400)

    :param output: The numpy output array from the model
    :type output: np.ndarray
    :return: The bounding boxes, scores and class ids
    """
    boxes: np.ndarray = np.array([], dtype=np.float32)
    scores: np.ndarray = np.array([], dtype=np.float32)
    class_ids: np.ndarray = np.array([], dtype=np.int32)
    # (1, 84, 8400) -> (8400, 84)
    logger.debug(f"{LP} output shape (1, <X>, 8400) detected!")
    predictions = np.squeeze(output[0]).T
    # logger.debug(f"{LP} predictions.shape = {predictions.shape} --- {predictions =}")
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    if len(scores) == 0:
        return_empty = True
    # Get bounding boxes for each object
    boxes = xywh2xyxy(predictions[:, :4])
    # Get the class ids with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    return boxes, scores, class_ids
