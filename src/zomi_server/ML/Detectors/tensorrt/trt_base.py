from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple, Union

from ....app import get_global_config

# Need the logger to log import errors
from ....Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)
LP: str = "TRT:cudart:"

try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = None
    logger.error(f"{LP} TensorRT not found, cannot use TensorRT detectors")

try:
    from cuda import cudart
except ModuleNotFoundError:
    cudart = None
    logger.error("CUDA python library not found, cannot use TensorRT detectors")

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
    logger.error("OpenCV not found, cannot use TensorRT detectors")

import numpy as np

from ....Models.Enums import ModelProcessor
from ....Models.config import (
    DetectionResults,
    Result,
    TRTModelConfig,
    TRTModelOptions,
    GlobalConfig,
)


g: Optional[GlobalConfig] = None
TRT_PLUGINS_LOCK = None

if trt is not None:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg, **kwargs):
            logger.log(self.get_severity(severity), msg, stacklevel=2)

        @staticmethod
        def get_severity(sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


def blob(im: np.ndarray, return_seg: bool = False) -> Union[np.ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def letterbox(
    im: np.ndarray,
    new_shape: Union[Tuple, List] = (640, 640),
    color: Union[Tuple, List] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def _postprocess(
    data: Tuple[np.ndarray],
    shape: Union[Tuple, List],
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs = data[0]
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0, 0, 0), dtype=np.int32),
        )

    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], 1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split(".")[:2])
    assert v0 == 4, "OpenCV version is wrong"
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres, iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )

    # divide each score in scores by 100 using matrix multiplication
    scores = scores / 100

    return bboxes, scores, labels


@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: Tuple
    cpu: np.ndarray
    gpu: int


class TensorRtDetector:
    scale: Optional[float] = None
    def __init__(self, model_config: TRTModelConfig) -> None:
        global g
        if g is None:
            g = get_global_config()
        assert model_config, f"{LP} no config passed!"

        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0, f"{LP} failed to create cuda stream"
        self.trt_logger = TrtLogger()
        self.config: TRTModelConfig = model_config
        self.options: TRTModelOptions = self.config.detection_options
        # hard code GPU as TRT is Nvidia GPUs only.
        self.processor: ModelProcessor = ModelProcessor.GPU
        self.name: str = self.config.name
        self.runtime_engine: Optional[trt.ICudaEngine] = None
        self.id: uuid.uuid4 = self.config.id
        self.description: Optional[str] = self.config.description
        self.LP: str = LP
        self.__init_engine()
        self.__init_bindings()
        # self.__warm_up()

    def __init_engine(self) -> None:
        global TRT_PLUGINS_LOCK

        logger.debug(
            f"{LP} initializing TensorRT engine: '{self.name}' ({self.id}) -- filename: {self.config.input.name}"
        )
        # Load plugins only 1 time
        if TRT_PLUGINS_LOCK is None:
            TRT_PLUGINS_LOCK = True
            logger.debug(f"{LP} initializing TensorRT plugins, should only happen once")
            trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        try:
            # De Serialize Engine
            with trt.Runtime(self.trt_logger) as runtime:
                self.runtime_engine = runtime.deserialize_cuda_engine(
                    self.config.input.read_bytes()
                )
            ctx = self.runtime_engine.create_execution_context()
        except Exception as ex:
            logger.error(f"{LP} {ex}", exc_info=True)
            raise ex
        else:
            # Grab and parse the names of the input and output bindings
            names = []
            self.num_bindings = self.runtime_engine.num_bindings
            self.bindings: List[int] = [0] * self.num_bindings
            num_inputs, num_outputs = 0, 0

            for i in range(self.runtime_engine.num_bindings):
                if self.runtime_engine.binding_is_input(i):
                    num_inputs += 1
                else:
                    num_outputs += 1
                names.append(self.runtime_engine.get_binding_name(i))
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            # set the context
            self.context = ctx
            self.input_names = names[:num_inputs]
            self.output_names = names[num_inputs:]
            logger.debug(
                f"{LP} input names ({len(self.input_names)}): {self.input_names} "
                f"// output names ({len(self.output_names)}): {self.output_names}"
            )

    def __init_bindings(self) -> None:
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        _start = time.time()
        logger.debug(f"{LP} initializing input/output bindings")
        for i, name in enumerate(self.input_names):
            assert (
                self.runtime_engine.get_binding_name(i) == name
            ), f"{LP} binding name mismatch"
            dtype = trt.nptype(self.runtime_engine.get_binding_dtype(i))
            shape = tuple(self.runtime_engine.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                # set model input size
                self.input_height = shape[2]
                self.input_width = shape[3]
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0, f"{LP} failed to allocate memory on GPU"
                # copy the data from the cpu to the gpu
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert (
                self.runtime_engine.get_binding_name(i) == name
            ), f"{LP} binding name mismatch"
            dtype = trt.nptype(self.runtime_engine.get_binding_dtype(i))
            shape = tuple(self.runtime_engine.get_binding_shape(i))
            #
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs
        logger.debug(
            f"{LP} initialized input/output bindings in {time.time() - _start:.3f} s. Dynamic input?: {dynamic}"
        )

    def warm_up(self) -> None:
        if self.is_dynamic:
            logger.warning(
                f"{LP} the engine ({self.config.input}) has dynamic axes, you are responsible for warming "
                f"up the engine"
            )
            return
        logger.debug(f"{LP}{self.name}: warming up the engine with 2 iterations...")
        _start = time.time()
        for _ in range(2):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)
        logger.debug(
            f"perf:{LP}{self.name}: engine warmed up in {time.time() - _start:.5f} seconds"
        )

    def resize_and_pad(self, bgr_image: np.ndarray, model_wxh: tuple, pad_color=(0, 0, 0)) -> np.ndarray:
        """
        Resize and pad an image to fit within model_wxh while maintaining aspect ratio.

        :param bgr_image: Input image as a BGR NumPy array (H, W, C).
        :param model_wxh: Target (width, height) for the model.
        :param pad_color: Padding color as (B, G, R), default is black.
        :return: Resized and padded image.
        """
        model_w, model_h = model_wxh
        h, w = bgr_image.shape[:2]
        # check if we need to resize at all
        if w == model_w and h == model_h:
            logger.debug(f"{LP} image already at model size, no need to resize")
            return bgr_image

        # Determine the appropriate scaling factor (maintaining aspect ratio)
        self.scale  = min(model_w / w, model_h / h)
        new_w, new_h = int(w * self.scale), int(h * self.scale)


        # Resize image while keeping aspect ratio
        resized_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ## create a black image and then overlay the resized image onto it, without centering
        padded_image = np.full((model_h, model_w, 3), pad_color, dtype=np.uint8)
        padded_image[:new_h, :new_w] = resized_image
        logger.debug(f"{LP} resized and padded image, original shape: {bgr_image.shape}, new shape: {padded_image.shape} // scale: {self.scale}")
        return padded_image

    async def detect(self, input_image: np.ndarray) -> DetectionResults:
        labels = np.array([], dtype=np.int32)
        confs = np.array([], dtype=np.float32)
        b_boxes = np.array([], dtype=np.float32)
        # bgr, ratio, dwdh = letterbox(input_image, self.inp_info[0].shape[-2:])
        # dw, dh = int(dwdh[0]), int(dwdh[1])

        self.img_height, self.img_width = input_image.shape[:2]
        # resize image to network size
        # rgb = cv2.resize(rgb, (self.input_width, self.input_height))
        proc_inp_img = self.resize_and_pad(input_image, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(proc_inp_img, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        # dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        detection_timer = time.time()
        try:
            async with get_global_config().async_locks.get(self.processor):
                data = self.__call__(tensor)
        except Exception as all_ex:
            logger.error(f"{LP} EXCEPTION! {all_ex}", exc_info=True)

            return DetectionResults(
                success=False,
                type=self.config.type_of,
                processor=self.processor,
                name=self.name,
                results=[],
            )
        else:
            logger.debug(
                f"perf:{LP} '{self.name}' inference took {time.time() - detection_timer:.5f} seconds"
            )
        b_boxes, confs, labels = self.process_output(list(data))
        # logger.debug(f"{LP} {lbls = } -- {confs = } -- {b_boxes = }")
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
                    lbl_idx: int = labels[i]
                    _label = self.config.labels[lbl_idx]
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
                    label=self.config.labels[labels[i]],
                    confidence=confs[i],
                    bounding_box=b_boxes[i],
                    color=colors,
                )
                for i in range(len(labels))
            ],
        )

        return result

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = (
            profiler
            if (profiler is not None and isinstance(profiler, trt.IProfiler))
            else trt.Profiler()
        )

    def __call__(self, *inputs) -> Union[Tuple, np.ndarray]:
        assert len(inputs) == self.num_inputs, f"{LP} incorrect number of inputs"
        contiguous_inputs: List[np.ndarray] = [np.ascontiguousarray(i) for i in inputs]
        for i in range(self.num_inputs):
            if self.is_dynamic:
                logger.debug(f"{LP} setting binding shape for input layer: {i}")
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(
                    contiguous_inputs[i].nbytes, self.stream
                )
                assert (
                    status.value == 0
                ), f"{LP} failed to allocate memory on GPU for dynamic input"
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu,
                contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            # copy the gpu pointer to the bindings
            self.bindings[i] = self.inp_info[i].gpu
        output_gpu_ptrs: List[int] = []
        outputs: List[np.ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert (
                    status.value == 0
                ), f"{LP} failed to allocate memory on GPU from dynamic input"
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = gpu
        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(output_gpu_ptrs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data,
                o,
                outputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        # for output in outputs:
        #     if isinstance(output, np.ndarray):
        #         logger.debug(f"{LP} {output.shape = }")

        # return tuple(outputs) if len(outputs) > 1 else outputs[0]
        return outputs

    def process_output(
        self, output: List[Optional[np.ndarray]]
    ) -> Tuple[List, List, List]:
        return_empty: bool = False
        boxes: np.ndarray = np.array([], dtype=np.float32)
        _boxes: np.ndarray = np.array([], dtype=np.float32)
        scores: np.ndarray = np.array([], dtype=np.float32)
        class_ids: np.ndarray = np.array([], dtype=np.int32)
        if output:
            output_type = self.config.output_type
            num_outputs = len(output)
            output_shape = [o.shape for o in output]
            logger.debug(f"{LP} '{self.name}' output_type: '{output_type.value}' / shapes ({num_outputs})"
                         f": {output_shape}")
            if not output_type:
                logger.error(f"{LP} '{self.name}' no output_type defined, cannot process model output into "
                             f"confidences, bounding boxes and class ids!")
                return_empty = True

            else:
                # yolov 8/11 pretrained: (1, 84, 8400)
                # yolov 10 pretrained: (1, 300, 6)
                # yolo-nas: depends on legacy or new flat/batched
                # FLAT (n, 7) and BATCHED outputs
                if output_type.value == "yolov8":
                    from ..process_output.yolov8 import process_output
                    _boxes, scores, class_ids = process_output(output[0])

                elif output_type.value == "yolonas":
                    from ..process_output.yolo_nas import process_output
                    _boxes, scores, class_ids = process_output(output)

                elif output_type.value == "yolov10":
                    from ..process_output.yolov10 import process_output
                    _boxes, scores, class_ids = process_output(output[0])

        else:
            logger.critical(f"{LP} '{self.name}' no output from model detected!")
            return_empty = True

        if not len(scores):
            return_empty = True

        if return_empty:
            logger.warning(f"{LP} '{self.name}' return_empty = True !!")
            return [], [], []

        # rescale boxes
        boxes = self.rescale_boxes(_boxes)
        do_nms = True
        if do_nms is True:
            nms, conf = self.options.nms.threshold, self.options.confidence
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)
            if len(indices) == 0:
                logger.debug(
                    f"{LP} no detections after filter by NMS ({nms}) and confidence ({conf})"
                )
                return [], [], []

            boxes = boxes[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
            if isinstance(boxes, tuple):
                if len(boxes) == 1:
                    boxes = boxes[0]
            if isinstance(scores, tuple):
                if len(scores) == 1:
                    scores = scores[0]

        return (
            boxes.astype(np.int32).tolist(),
            scores.astype(np.float32).tolist(),
            class_ids.astype(np.int32).tolist(),
        )

    def rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions using the formula from resize_and_pad."""
        if self.scale is None:
            raise ValueError("Scale factor is not set. Ensure resize_and_pad() has been called.")

        # Rescale boxes to the original image dimensions
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.scale  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.scale  # y1, y2

        return boxes


    def old_rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        # take into account the scaling formula used in the resize_and_pad method


        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes
