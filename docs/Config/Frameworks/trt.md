# `trt` (Tensor RT) model config

## `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `trt` `framework` models.

## `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!TIP]
> If the `classes` key is omitted, the COCO 2017 classes (80 labels) are used.

## `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

## `processor`
- `processor: <string>`
- Default: `gpu`

The `processor` key is used to set the processor to use for that model. It will always be set to `gpu`

## `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device:
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

## `output_type`
- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server
how to process the output from the model into confidence, bounding box, and class id.

## `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

#### `threshold`
- `threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.35`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more
overlapping bounding boxes out.

## Example
```yaml
models:
# TensorRT Model Example (User must install TensorRT and compile their engine model)
  - name: yolo-nas-s trt
    enabled: no
    description: "TensorRT optimized YOLO-NAS-S pretrained model"
    input: "/shared/models/yolo/yolo_nas_s.trt"
    #gpu_idx: 0
    framework: trt

    type_of: object
    # - Only ort, torch and trt support output_type
    # - Tells the server how to process the output from the model into confidence, bounding box, and class id
    output_type: yolonas

    # - Make sure to set the proper model input size or it will throw errors!
    height: 640
    width: 640

    detection_options:
      confidence: 0.31
      # ort and trt support nms enabled/disabled (YOLO v10 does not need NMS)
      nms:
        enabled: yes
        threshold: 0.44
```