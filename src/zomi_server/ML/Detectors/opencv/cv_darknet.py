from __future__ import annotations
import time
from functools import lru_cache
from logging import getLogger
from typing import Optional, TYPE_CHECKING, List
from warnings import warn

try:
    import cv2
except ImportError:
    warn("OpenCV not installed, cannot use OpenCV detectors")
    raise
import numpy as np

from ....Models.Enums import ModelProcessor
from .cv_base import CV2Base
from ....Models.config import DetectionResults, Result
from ....Log import SERVER_LOGGER_NAME

if TYPE_CHECKING:
    from ....Models.config import CV2YOLOModelConfig

LP: str = "OpenCV:DarkNet:"
logger = getLogger(SERVER_LOGGER_NAME)
# TODO: Choose what models to load and keep in memory and which to load and drop for memory constraints


class CV2DarkNetDetector(CV2Base):
    _classes: Optional[List] = None
    lp = LP

    def __init__(self, model_config: CV2YOLOModelConfig):
        super().__init__(model_config)
        self.gpu_idx = self.config.gpu_idx
        if self.gpu_idx is None and self.processor == ModelProcessor.GPU:
            logger.debug(f"{self.lp} GPU index not set, defaulting to 0")
            self.gpu_idx = 0
        # Model init params not initialized in super()
        self.model: Optional[cv2.dnn.DetectionModel] = None
        # logger.debug(f"{self.lp} configuration: {self.config}")
        self.load_model()

    @property
    @lru_cache
    def classes(self):
        if self._classes is None:
            self._classes = self.config.labels
        return self._classes

    @classes.setter
    def classes(self, value: List):
        self._classes = value

    def load_model(self):
        logger.debug(
            f"{self.lp} loading model into processor [{self.processor}] memory: {self.name} ({self.id})"
        )
        load_timer = time.time()
        try:
            # Allow for .weights/.cfg and .onnx YOLO architectures
            model_file: str = self.config.input.as_posix()
            config_file: Optional[str] = None
            if self.config.config:
                if self.config.config.exists():
                    config_file = self.config.config.as_posix()
                else:
                    raise FileNotFoundError(
                        f"{self.lp} config file '{self.config.config}' not found!"
                    )
            logger.info(
                f"{self.lp} loading -> model: {model_file} // config: {config_file}"
            )
            self.net = cv2.dnn.readNet(model_file, config_file)
        except Exception as model_load_exc:
            logger.error(
                f"{self.lp} Error while loading model file and/or config! "
                f"(May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise model_load_exc
        # DetectionModel allows to set params for preprocessing input image. DetectionModel creates net
        # from file with trained weights and config, sets preprocessing input, runs forward pass and return
        # result detections. For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.
        if self.net is not None:
            self.model = cv2.dnn.DetectionModel(self.net)
            self.model.setInputParams(
                scale=1 / 255, size=(self.config.width, self.config.height), swapRB=True
            )
            self.cv2_processor_check()
            if self.processor == ModelProcessor.GPU:
                logger.debug(f"{self.lp} set CUDA/cuDNN backend and target")
                cuda_devs = cv2.cuda.getCudaEnabledDeviceCount()
                choices = [x for x in range(cuda_devs)]
                if self.gpu_idx in range(cuda_devs):
                    logger.debug(
                        f"{self.lp} Using GPU {cv2.cuda.printShortCudaDeviceInfo(self.gpu_idx)}"
                    )
                    cv2.cuda.setDevice(self.gpu_idx)
                else:
                    logger.warning(
                        f"{self.lp} Invalid CUDA GPU index configured: {self.gpu_idx}"
                    )
                    if cuda_devs > 1:
                        logger.info(f"{self.lp} Valid options for GPU are {choices}:")
                        for cuda_device in range(cuda_devs):
                            cv2.cuda.printShortCudaDeviceInfo(cuda_device)

            logger.debug(
                f"perf:{self.lp} '{self.name}' loading completed in {time.time() - load_timer:.5f} s"
            )
        else:
            logger.debug(
                f"perf:{self.lp} '{self.name}' FAILED in {time.time() - load_timer:.5f} s"
            )

    async def detect(self, input_image: np.ndarray):
        result = []
        if input_image is None:
            raise ValueError(f"{self.lp} no image passed!")
        if not self.net:
            self.load_model()
        _h, _w = self.config.height, self.config.width
        if self.config.square:
            input_image = self.square_image(input_image)
        h, w = input_image.shape[:2]
        # dnn.DetectionModel resizes the image and calculates scale of bounding boxes for us
        labels, confs, b_boxes = [], [], []
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence

        logger.debug(
            f"{self.lp}detect: '{self.name}' ({self.processor}) - "
            f"input image {w}*{h} - model input {_w}*{_h}"
            f"{' [squared]' if self.config.square else ''}"
        )

        try:
            detection_timer = time.time()
            from ....app import get_global_config

            async with get_global_config().async_locks.get(self.processor):
                l, c, b = self.model.detect(input_image, conf_threshold, nms_threshold)

            for class_id, confidence, box in zip(l, c, b):
                confidence = float(confidence)
                x, y, _w, _h = (
                    int(round(box[0])),
                    int(round(box[1])),
                    int(round(box[2])),
                    int(round(box[3])),
                )
                b_boxes.append(
                    [
                        x,
                        y,
                        x + _w,
                        y + _h,
                    ]
                )
                labels.append(self.config.labels[class_id])
                confs.append(confidence)
        except Exception as all_ex:
            err_msg = repr(all_ex)
            # cv2.error: OpenCV(4.2.0) /home/<Someone>/opencv/modules/dnn/src/cuda/execution.hpp:52: error: (-217:Gpu
            # API call) invalid device function in function 'make_policy'
            logger.error(f"{self.lp} exception during detection -> {all_ex}")
            # OpenCV 4.7.0 Weird Error fixed with rolling fix
            # OpenCV:YOLO: exception during detection -> OpenCV(4.7.0-dev) /opt/opencv/
            # modules/dnn/src/layers/cpu_kernels/conv_winograd_f63.cpp:401: error: (-215:Assertion failed) CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4 && CONV_WINO_ATOM_F32 == 4 in function 'winofunc_BtXB_8x8_f32'
            if err_msg.find("-215:Assertion failed") > 0:
                if err_msg.find("CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4") > 0:
                    _msg = (
                        f"{self.lp} OpenCV 4.7.x WEIRD bug detected! "
                        f"Please update to OpenCV 4.7.1+ or 4.6.0 or less!"
                    )
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            elif err_msg.find("-217:Gpu") > 0:
                if (
                    err_msg.find("'make_policy'") > 0
                    and self.processor == ModelProcessor.GPU
                ):
                    _msg = (
                        f"{self.lp} (-217:Gpu # API call) invalid device function in function 'make_policy' - "
                        f"This happens when OpenCV is compiled with the incorrect Compute Capability "
                        f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with "
                        f"the correct CUDA_ARCH_BIN before GPU detections will work properly!"
                    )
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            raise all_ex
        logger.debug(
            f"perf:{self.lp}{self.processor}: '{self.name}' detection "
            f"took: {time.time() - detection_timer:.5f} s"
        )
        colors: Optional[dict] = None
        clr_cfg = get_global_config().config.color
        if clr_cfg and clr_cfg.enabled:
            if b_boxes:
                # crop each bbox and send to color detection
                inc_labels: Optional[List[str]] = (
                    clr_cfg.labels if clr_cfg.labels else None
                )
                color_start = time.time()
                for i, bbox in enumerate(b_boxes):
                    _label = labels[i]
                    # if labels isnt set in color config, run on all labels.
                    # if labels are defined in color config, only run on labels in the list.
                    if inc_labels and _label not in inc_labels:
                        logger.warning(
                            f"{LP} '{self.name}' label '{_label}' not in color detection labels, skipping..."
                        )
                        continue
                    x, y, x2, y2 = bbox
                    # crop the image
                    crop = input_image[y:y2, x:x2]
                    # send to color detection
                    colors = await get_global_config().color_detector.detect(crop)
                logger.debug(
                    f"perf:{LP} Color detection took {time.time() - color_start:.5f} seconds "
                    f"(total: {time.time() - detection_timer:.5f})"
                )
        result = DetectionResults(
            success=True if labels else False,
            type=self.config.type_of,
            processor=self.processor,
            name=self.name,
            results=[
                Result(
                    label=labels[i],
                    confidence=confs[i],
                    bounding_box=b_boxes[i],
                    color=colors,
                )
                for i in range(len(labels))
            ],
        )

        return result
