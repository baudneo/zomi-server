import logging
from typing import List, Tuple

import numpy as np

from ....Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)
LP: str = "YOLO-NAS:"


def process_output(output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the output from the YOLO v10 pretrained models (1, 300, 6)

    :param output: The numpy output array from the model
    """

    pass