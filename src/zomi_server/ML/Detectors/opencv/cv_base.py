from __future__ import annotations

import time
from logging import getLogger
from typing import Optional, Union, TYPE_CHECKING

import cv2
import numpy as np
from ....Log import SERVER_LOGGER_NAME

from ....Models.Enums import ModelProcessor

if TYPE_CHECKING:
    from ....Models.config import (
        BaseModelOptions,
        CV2YOLOModelOptions,
        CV2YOLOModelConfig,
        CV2HOGModelConfig,
    )


logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "OpenCV DNN:"


class CV2Base:
    lp = LP

    def warm_up(self):
        """Warm up the YOLO model by running a dummy inference."""
        runs = 2
        if self.model:
            logger.debug(f"{self.lp}{self.name}: warming up the model with {runs} iterations...")
            # create a random image to warm up YOLO
            _h, _w = self.config.height, self.config.width
            image = np.random.randint(0, 256, (_h, _w, 3), dtype=np.uint8)
            warmup_start = time.time()
            for _ in range(runs):
                _ = self.model.detect(image, 0.3, 0.3)
            logger.debug(
                f"perf:{self.lp}{self.name}: model warmed up in {time.time() - warmup_start:.5f} seconds"
            )

        else:
            logger.error(f"{self.lp} model not initialized, cannot warm up!")

    def __init__(
        self,
        model_config: Union[CV2YOLOModelConfig, CV2HOGModelConfig],
    ):
        if not model_config:
            raise ValueError(f"{self.lp} no config passed!")
        # Model init params
        self.config = model_config
        self.options: Union[CV2YOLOModelOptions, BaseModelOptions] = (
            self.config.detection_options
        )
        self.processor: ModelProcessor = self.config.processor
        self.name = self.config.name
        self.net: Optional[cv2.dnn] = None
        self.model = None
        self.id = self.config.id
        self.description = self.config.description

    def square_image(self, frame: np.ndarray):
        """Zero pad the matrix to make the image squared"""
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        logger.debug(
            f"{self.lp}squaring image-> '{self.name}' before padding: {frame.shape} - after padding: {result.shape}"
        )
        return result

    @staticmethod
    def cv2_version() -> int:
        _maj, _min, _patch = "", "", ""
        x = cv2.__version__.split(".")
        x_len = len(x)
        if x_len <= 2:
            _maj, _min = x
            _patch = "0"
        elif x_len == 3:
            _maj, _min, _patch = x
            _patch = _patch.replace("-dev", "") or "0"
        else:
            logger.error(f'come and fix me again, cv2.__version__.split(".")={x}')
        return int(_maj + _min + _patch)

    def cv2_processor_check(self):
        if self.config.processor == ModelProcessor.GPU:
            logger.debug(
                f"{self.lp} '{self.name}' GPU configured as the processor, running checks..."
            )
            cv_ver = self.cv2_version()
            if cv_ver < 420:
                logger.error(
                    f"{self.lp} '{self.name}' You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum"
                    f" of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7"
                    f" on how to compile and install OpenCV with CUDA"
                )
                self.processor = self.config.processor = ModelProcessor.CPU
            else:  # Passed opencv version check, using GPU
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                if self.config.cv2_cuda_fp_16:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    logger.debug(
                        f"{self.lp} '{self.name}' half precision floating point (FP16) cuDNN target enabled (turn this "
                        f"off if it makes detections slower or you see 'NaN' errors!)"
                    )
                else:
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif self.config.processor == ModelProcessor.CPU:
            logger.debug(f"{self.lp} '{self.name}' CPU configured as the processor")
