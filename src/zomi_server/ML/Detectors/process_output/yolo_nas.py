import logging
from typing import List, Tuple

try:
    import cupy as cp  # Try importing CuPy
    _HAS_CUPY = True
    np = cp  # Dynamically alias CuPy as `np`
except ImportError:
    import numpy as np  # Fallback to NumPy
    _HAS_CUPY = False

from ....Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)
LP: str = "YOLO-NAS:"


def process_output(output: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the output from the YOLO-NAS pretrained models

    :param output: List of numpy output arrays (2 or 4) from the model
    :return: Tuple of the bounding boxes, scores and class ids
    """
    boxes: np.ndarray = np.array([], dtype=np.float32)
    scores: np.ndarray = np.array([], dtype=np.float32)
    class_ids: np.ndarray = np.array([], dtype=np.int32)
    output_shape = [o.shape for o in output]
    legacy_shape = [(1, 8400, 4), (1, 8400, 80)]
    new_flat_shape = [(1, 8400, 80), (1, 8400, 4)]
    num_outputs = len(output)
    # Legacy / Flat output
    if num_outputs == 2:
        if output_shape == legacy_shape or output_shape == new_flat_shape:
            # NAS - .convert_to_onnx() output = [(1, 8400, 4), (1, 8400, 80)]
            # NEW "FLAT" = [(1, 8400, 80), (1, 8400, 4)]
            raw_scores: np.ndarray
            if output_shape == legacy_shape:
                shape_str = "convert_to_onnx() [Legacy]"
                _boxes, raw_scores = output
            elif output_shape == new_flat_shape:
                shape_str = "export() [NEW Flat]"
                raw_scores, _boxes = output
            else:
                shape_str = "<UNKNOWN>"
                _boxes, raw_scores = output
            logger.debug(f"{LP} model.{shape_str} output detected!")

            # find max from scores and flatten it [1, n, num_class] => [n]
            scores = raw_scores.max(axis=2).flatten()
            # squeeze boxes [1, n, 4] => [n, 4]
            boxes = np.squeeze(_boxes, 0)
            # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
            class_ids = np.argmax(raw_scores, axis=2).flatten()
        else:
            logger.warning(
                f"{LP} unknown output shape: {output_shape}, should be"
                f" Legacy: {legacy_shape} or New Flat: {new_flat_shape}"
            )
    # NAS model.export() batch output
    elif num_outputs == 4:
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

    return boxes, scores, class_ids
