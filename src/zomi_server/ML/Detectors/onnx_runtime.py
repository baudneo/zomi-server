from __future__ import annotations

import asyncio
import grp
import os
import pwd
import re
import time
import uuid
from logging import getLogger
from pathlib import Path
from pprint import pformat
from typing import (
    Optional,
    TYPE_CHECKING,
    List,
    Tuple,
    Union,
    AnyStr,
    Dict,
    Any,
)
from warnings import warn

from pydantic import Field
from pydantic.dataclasses import dataclass

try:
    import cv2
except ImportError:
    warn("OpenCV not installed! Please install/link to it!")

    raise
try:
    import onnxruntime as ort
except ImportError:
    warn("onnxruntime not installed, cannot use onnxruntime detectors")
    ort = None
import numpy as np

from ...app import get_global_config
from ...Models.Enums import ModelProcessor
from ...Models.config import DetectionResults, Result
from ...Log import SERVER_LOGGER_NAME
from .ocr import OCRBase

if TYPE_CHECKING:
    from ...Models.config import ORTModelConfig, ORTModelOptions

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "ORT:"


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # logger.debug(f"{LP} {keep_indices.shape = }, {sorted_indices.shape = }")
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class ORTDetector(OCRBase):
    def __init__(self, model_config: ORTModelConfig):
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        self.config = model_config
        self.options: ORTModelOptions = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name: str = self.config.name
        self.model: Optional[str] = None
        self.id: uuid.uuid4 = self.config.id
        self.description: Optional[str] = model_config.description
        self.session: Optional[ort.InferenceSession] = None
        self.img_height: int = 0
        self.img_width: int = 0
        self.input_names: List[Optional[AnyStr]] = []
        self.LP: str = f"{LP}'{self.name}':"

        self.input_shape: Tuple[int, int, int] = (0, 0, 0)
        self.input_height: int = 0
        self.input_width: int = 0

        self.output_names: Optional[List[Optional[AnyStr]]] = None
        self.initialize_model(self.config.input)
        if self.config.ocr:
            super().__init__(self.config.ocr)

    def __call__(self, image):
        return self.detect(image)

    def initialize_model(self, path: Path):
        logger.debug(
            f"{LP} loading model into processor [{self.processor}] memory: {self.name} ({self.id})"
        )
        providers = ["CPUExecutionProvider"]
        # Check if GPU is available
        if self.processor == ModelProcessor.GPU:
            available_providers = ort.get_available_providers()

            gpu_idx = self.config.gpu_idx
            if gpu_idx is None:
                gpu_idx = 0

            if ort.get_device() == "GPU":
                if "CUDAExecutionProvider" in available_providers:
                    providers.insert(
                        0,
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": gpu_idx,
                                # 'arena_extend_strategy': 'kNextPowerOfTwo',
                                # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                                # 'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                # 'do_copy_in_default_stream': True,
                            },
                        ),
                    )

                if "ROCMExecutionProvider" in available_providers:
                    # get execution providers
                    providers.insert(1, "ROCMExecutionProvider")

            else:
                logger.warning(
                    f"{LP} GPU not available, using CPU for model: {self.name}"
                )
                self.processor = self.config.processor = ModelProcessor.CPU
        self.session = ort.InferenceSession(path, providers=providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def warm_up(self):
        """Warm up model"""
        logger.debug(f"{LP} Warming up model: {self.name}")
        # reverse h/w
        input_tensor = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(1, self.input_shape[1], self.input_shape[3], self.input_shape[2]),
        ).astype(np.float32)
        # execute inference on input tensor asynchronously
        asyncio.run(self.inference(input_tensor))

    async def detect(self, image: np.ndarray):
        b_boxes: List
        confs: List
        labels: List
        input_tensor = self.prepare_input(image)
        logger.debug(
            f"{LP}detect: '{self.name}' ({self.processor}) - "
            f"input image {self.img_width}*{self.img_height} - model input {self.config.width}*{self.config.height}"
            f"{' [squared]' if self.config.square else ''}"
        )
        detection_timer = time.time()
        outputs = await self.inference(input_tensor)
        b_boxes, confs, labels = self.process_output(outputs)
        logger.debug(
            f"perf:{LP}{self.processor}: '{self.name}' detection "
            f"took: {time.time() - detection_timer:.5f} seconds"
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
                    crop = image[y:y2, x:x2]
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

        if self.ocr_readers and b_boxes:

            @dataclass
            class OCRResult:
                text: str
                confs: List[float] = Field(default_factory=list)
                coords: List[List[List[int]]] = Field(default_factory=list)
                hits: float = 0.0

                avg_conf: Optional[float] = None
                high_conf: Optional[float] = None

            def _proc_ocr_result(text: str, conf: float, coords: List[List[int, int]]):
                if text:
                    # normalize text by removing non-alphanumeric characters and converting to upper case
                    text = text.upper()
                    text = re.sub(r"[^A-Z0-9]", "", text)
                    # Do regex pattern matching to ascertain if the text conforms to a plate number
                    # if so, add it to the final results

                    if text not in final_results:
                        logger.debug(
                            f"Creating OCRResult dataclass with text: {text} - confs: {[confs]} -- coords {[coords]} -- hits: {1.0}"
                        )
                        final_results[text] = OCRResult(
                            text=text,
                            confs=[conf],
                            coords=[coords],
                            hits=1.0,
                        )
                    else:
                        final_results[text].hits += 1
                        final_results[text].coords.append(coords)
                        final_results[text].confs.append(confidence)

            best_results: List = []
            final_results: Dict[str, OCRResult] = {}
            final_results: List[Tuple[str, float, int]] = []
            logger.debug(f"{LP}detect: '{self.name}' - running OCR")
            if self.ocr_cfg.debug_images:
                # check if directory exists, if not, create it
                img_dir = self.ocr_cfg.dbg_img_dir
                if not img_dir:
                    import tempfile

                    img_dir = Path(tempfile.gettempdir()) / "zomi_server/server/ocr"
                    logger.warning(
                        f"{LP}OCR: No debug image directory specified, using default: {img_dir}"
                    )
                if img_dir and isinstance(img_dir, Path):
                    if not img_dir.exists():
                        logger.debug(
                            f"{LP}OCR: Creating directory for debug images: {img_dir}"
                        )
                        img_dir.mkdir(parents=True, exist_ok=True)
                    files = img_dir.glob("*.jpg")
                    if files:
                        logger.debug(
                            f"OCR DEBUG>>> removing existing files in '{img_dir}'"
                        )
                        for f in files:
                            os.remove(f)
                    self.ocr_cfg.dbg_img_dir = img_dir
                else:
                    logger.warning(
                        f"{LP}OCR: debug image directory is not a Path object, cannot save debug images. "
                        f"Check your config."
                    )
            _conf = self.ocr_cfg.confidence
            if not _conf:
                _conf = 0.5
            sim_thresh = self.ocr_cfg.word_similarity_thresh
            if sim_thresh is None:
                sim_thresh = 3
            # use same percent for x,y to retain aspect ratio (aspect can be used to filter contours)
            crop_leeway_percent = self.ocr_cfg.crop_leeway_perc
            if crop_leeway_percent is None:
                crop_leeway_percent = 0.025
                self.ocr_cfg.crop_leeway_perc = crop_leeway_percent

            # Todo: base the crop pixels percent on the size of the cropped object instead of image
            crop_leeway_w = int(self.img_width * crop_leeway_percent)
            crop_leeway_h = int(self.img_height * crop_leeway_percent)

            logger.debug(
                f"{LP}detect: '{self.name}' - crop pixels leeway ({crop_leeway_percent * 100:.3f}%): "
                f"{crop_leeway_w} based on width: {self.img_width}"
            )

            for idx, (cx1, cy1, cx2, cy2) in enumerate(b_boxes):
                # Reset on each bbox
                final_results, final_results = [], {}

                # crop the bounding box from the image to pass to OCR
                # add 'leeway' pixels to each side to ensure we get the whole plate
                # check that the cropped image is not out of bounds, if so, correct it to only grab whats available
                cx1 -= crop_leeway_w
                cy1 -= crop_leeway_h
                cx2 += crop_leeway_w
                cy2 += crop_leeway_h
                # Adjust cropped coordinates
                crop_coords = [cx1, cy1, cx2, cy2]
                # Image bounds: [min_x, min_y, max_x, max_y]
                bounds = [0, 0, self.img_width, self.img_height]

                for i in range(2):
                    crop_coords[i] = max(crop_coords[i] - crop_leeway_w, bounds[i])
                    crop_coords[i + 2] = min(
                        crop_coords[i + 2] + crop_leeway_h, bounds[i + 2]
                    )

                cx1, cy1, cx2, cy2 = crop_coords
                cropped_image = image[cy1:cy2, cx1:cx2]

                ocr_results: Optional[Dict[str, Any]] = await self.ocr(
                    cropped_image, num=idx + 1
                )
                if ocr_results:
                    for ocr_eng, results in ocr_results.items():
                        if ocr_eng and results:
                            ocr_eng: str = ocr_eng.casefold()
                            results: List
                            if ocr_eng.startswith("easyocr_"):
                                for _res in results:
                                    _res: Tuple[List[List[int, int]], str, float]
                                    coords, text, confidence = _res
                                    if text:
                                        _proc_ocr_result(text, confidence, coords)

                            elif ocr_eng.casefold().startswith("paddleocr_"):
                                if results:
                                    if results[0]:
                                        for _result in results[0]:
                                            if _result:
                                                _result: List[
                                                    List[List[float, float]],
                                                    Tuple[str, float],
                                                ]

                                                coords, (text, confidence) = _result
                                                coords = np.array(
                                                    coords, dtype=np.int32
                                                ).tolist()
                                                if text:
                                                    _proc_ocr_result(
                                                        text, confidence, coords
                                                    )

                            elif ocr_eng.casefold().startswith("tesseract_"):
                                confidence = 0.65
                                coords = []
                                if text:
                                    _proc_ocr_result(text, confidence, coords)

                    # Work through word similarity
                    hit_high = 0
                    conf_high = 0.0
                    avg_conf_high = 0.0
                    hit_high_text = ""
                    conf_high_text = ""
                    avg_conf_high_text = ""
                    all_plates = (x for x in final_results.keys() if x)
                    proc_texts = []
                    if final_results:
                        logger.debug(
                            f"{LP}detect: '{self.name}' - using word similarity threshold "
                            f"of {sim_thresh} edit(s)"
                        )
                        for plate_number, ro in final_results.items():
                            if not ro:
                                continue
                            logger.debug(
                                f"Checking '{plate_number}' for word similarities"
                            )
                            proc_texts.append(plate_number)
                            confs = ro.confs
                            hits = ro.hits
                            avg_conf = sum(confs) / len(confs)
                            # sort confs by descending
                            confs = sorted(confs, reverse=True)
                            # set high_conf
                            high_conf = confs[0]
                            ro.high_conf = high_conf
                            ro.avg_conf = avg_conf
                            ro.confs = confs
                            for _word in all_plates:
                                if _word in proc_texts:
                                    continue
                                if self.check_similarity(
                                    plate_number, _word, sim_thresh
                                ):
                                    logger.debug(
                                        f"{LP}detect: '{self.name}' - word '{plate_number}' is similar to '{_word}'"
                                    )
                                    hits += 0.5
                                    final_results[_word].hits += 0.5

                            ro.hits = hits

                            if hits > hit_high:
                                hit_high = hits
                                hit_high_text = plate_number
                            if high_conf > conf_high:
                                conf_high = high_conf
                                conf_high_text = plate_number
                            if avg_conf > avg_conf_high:
                                avg_conf_high = avg_conf
                                avg_conf_high_text = plate_number

                            final_results[plate_number] = ro
                            logger.debug(f"Done checking '{plate_number}'")

                        # calculate the best result based on hits, avg_cong and high_conf
                        if hit_high_text and conf_high_text and avg_conf_high_text:
                            if hit_high_text == conf_high_text == avg_conf_high_text:
                                final_results[hit_high_text].hits += 3
                            elif hit_high_text == conf_high_text:
                                final_results[hit_high_text].hits += 2
                            elif hit_high_text == avg_conf_high_text:
                                final_results[hit_high_text].hits += 2
                            elif conf_high_text == avg_conf_high_text:
                                final_results[conf_high_text].hits += 2
                            else:
                                final_results[hit_high_text].hits += 1
                                final_results[conf_high_text].hits += 1
                                final_results[avg_conf_high_text].hits += 1

                        # sort fin_res dict by hits
                        final_results = dict(
                            sorted(
                                final_results.items(),
                                key=lambda x: x[1].hits,
                                reverse=True,
                            )
                        )
                        hit_high_text = list(final_results.keys())[0]
                        # get the best result (first key, value since its sorted)
                        best_result = final_results[hit_high_text]
                        avg = best_result.avg_conf
                        logger.debug(
                            f"SHORTENED FINAL RESULTS for object {idx+1}: {pformat(final_results, indent=4)}"
                        )

                        if avg > _conf:
                            logger.debug(
                                f"\n\n\n BEST RESULT for object {idx + 1} = '{hit_high_text}'\n"
                                f"{pformat(best_result, indent=6)} \n\n"
                            )
                            best_results.append((hit_high_text, avg, b_boxes[idx]))

            if best_results:
                # Add more results, not just top 1
                # best_results.append((hit_high_text, avg, b_boxes[idx]))

                result = DetectionResults(
                    success=True if labels else False,
                    name=self.name,
                    type=self.config.type_of,
                    processor=self.processor,
                    results=[
                        Result(
                            label=best_results[i][0],
                            confidence=best_results[i][1],
                            bounding_box=best_results[i][2],
                        )
                        for i in range(len(best_results))
                    ],
                    # extra_image_data={
                    #     "ocr": {
                    #         k: v
                    #         for k, v in final_results.items()
                    #         if v and k != best_result
                    #     }
                    },
                )

            if self.ocr_cfg.debug_images:
                uid = pwd.getpwnam("www-data").pw_uid
                gid = grp.getgrnam("www-data").gr_gid
                files = self.ocr_cfg.dbg_img_dir.glob("*.jpg")

                for f in files:
                    os.chown(f, uid, gid)

        return result

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare a numpy array image for onnxruntime InferenceSession"""
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config.square:
            logger.debug(f"{LP}detect: '{self.name}' - padding image to square")
            # Pad image to square
            input_img = np.pad(
                input_img,
                (
                    (0, max(self.img_height, self.img_width) - self.img_height),
                    (0, max(self.img_height, self.img_width) - self.img_width),
                    (0, 0),
                ),
                "constant",
                constant_values=0,
            )

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    async def inference(
        self, input_tensor
    ) -> Optional[List[Union[np.ndarray, List, Dict]]]:
        """Perform inference on the prepared input image"""
        logger.debug(f"{LP} '{self.name}' running inference")
        outputs: Optional[List[Union[np.ndarray, List, Dict]]] = None
        try:
            async with get_global_config().async_locks.get(self.processor):
                outputs = self.session.run(
                    self.output_names, {self.input_names[0]: input_tensor}
                )
        except Exception as e:
            logger.error(
                f"{LP} '{self.name}' Error while running model: {e}", exc_info=True
            )

        return outputs

    def process_output(
        self, output: List[Optional[np.ndarray]]
    ) -> Tuple[List, List, List]:
        return_empty: bool = False
        if output:
            logger.debug(
                f"{LP} '{self.name}' output shapes: {[o.shape for o in output]}"
            )
            # NAS new model.export() has FLAT (n, 7) and BATCHED  outputs
            num_outputs = len(output)
            if num_outputs == 1:
                if isinstance(output[0], np.ndarray):
                    # prettrained: (1, 84, 8400)
                    # dfo with 2 classes and 1 background: (1, 6, 8400)
                    if output[0].shape[0] == 1 and output[0].shape[2] == 8400:
                        # v8
                        # (1, 84, 8400) -> (8400, 84)
                        predictions = np.squeeze(output[0]).T
                        logger.debug(
                            f"{LP} yolov8 output shape = (1, <X>, 8400) detected!"
                        )
                        # Filter out object confidence scores below threshold
                        scores = np.max(predictions[:, 4:], axis=1)
                        # predictions = predictions[scores > self.options.confidence, :]
                        # scores = scores[scores > self.options.confidence]

                        if len(scores) == 0:
                            return_empty = True

                        # Get the class with the highest confidence
                        # Get bounding boxes for each object
                        boxes = self.extract_boxes(predictions)
                        class_ids = np.argmax(predictions[:, 4:], axis=1)
                    elif len(output[0].shape) == 2 and output[0].shape[1] == 7:
                        logger.debug(
                            f"{LP} YOLO-NAS model.export() FLAT output detected!"
                        )
                        # YLO-NAS .export FLAT output = (n, 7)
                        flat_predictions = output[0]
                        # pull out the class index and class score from the predictions
                        # and convert them to numpy arrays
                        flat_predictions = np.array(flat_predictions)
                        class_ids = flat_predictions[:, 6].astype(int)
                        scores = flat_predictions[:, 5]
                        # pull the boxes out of the predictions and convert them to a numpy array
                        boxes = flat_predictions[:, 1:4]
            elif num_outputs == 2:
                # NAS - .convert_to_onnx() output = [(1, 8400, 4), (1, 8400, 80)]
                if output[0].shape == (1, 8400, 4) and output[1].shape == (1, 8400, 80):
                    # YOLO-NAS
                    logger.debug(
                        f"{LP} YOLO-NAS model.convert_to_onnx() output detected!"
                    )
                    _boxes: np.ndarray
                    raw_scores: np.ndarray
                    # get boxes and scores from outputs
                    _boxes, raw_scores = output
                    # find max from scores and flatten it [1, n, num_class] => [n]
                    scores = raw_scores.max(axis=2).flatten()
                    if len(scores) == 0:
                        return_empty = True
                    # squeeze boxes [1, n, 4] => [n, 4]
                    _boxes = np.squeeze(_boxes, 0)
                    boxes = self.rescale_boxes(_boxes)
                    # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
                    class_ids = np.argmax(raw_scores, axis=2).flatten()
            elif num_outputs == 4:
                # NAS model.export() batch output len = 4
                # num_predictions [B, 1]
                # pred_boxes [B, N, 4]
                # pred_scores [B, N]
                # pred_classes [B, N]
                # Here B corresponds to batch size and N is the maximum number of detected objects per image
                if (
                    len(output[0].shape) == 2
                    and len(output[1].shape) == 3
                    and len(output[2].shape) == 2
                    and len(output[3].shape) == 2
                ):
                    logger.debug(
                        f"{LP} YOLO-NAS model.export() BATCHED output detected!"
                    )
                    batch_size = output[0].shape[0]
                    max_detections = output[1].shape[1]
                    num_predictions, pred_boxes, pred_scores, pred_classes = output
                    assert (
                        num_predictions.shape[0] == 1
                    ), "Only batch size of 1 is supported by this function"

                    num_predictions = int(num_predictions.item())
                    boxes = pred_boxes[0, :num_predictions]
                    scores = pred_scores[0, :num_predictions]
                    class_ids = pred_classes[0, :num_predictions]
        else:
            return_empty = True

        if return_empty:
            return [], [], []
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.options.confidence, self.options.nms
        )
        if len(indices) == 0:
            logger.debug(
                f"{LP} no detections after filter by NMS ({self.options.nms}) and confidence ({self.options.confidence})"
            )
            return [], [], []
        else:
            logger.debug(f"{LP} '{self.name}' NMS indices: {indices =}")
            boxes = (boxes[indices],)
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

    def extract_boxes(self, predictions):
        """Extract boxes from predictions, scale them and convert from xywh to xyxy format"""
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
