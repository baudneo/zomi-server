# `ort` (ONNXRuntime) model config
The `ort` `framework` is used for ONNX models that are supported by ONNXRuntime.

## `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device:
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

## `gpu_framework`
>[!WARNING]
> **This is a WIP feature and is not implemented yet.**

- `gpu_framework: <string>`
- `cuda` or `rocm` *WIP*

The `gpu_framework` key is used to set the GPU framework to use. Hopefully we get ROCm support for 
torch and onnxruntime (`ort`)

## `output_type`
- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server
how to process the output from the model into confidence, bounding box, and class id.

## `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

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
  - name: yolov8s onnx
    description: "Ultralytics YOLO v8s pretrained ONNX model on onnxruntime"
    enabled: no
    framework: ort
    type_of: object
    processor: gpu
    #gpu_idx: 0
    # - Possible ROCm support *WIP*
    #gpu_framework: cuda
    input: "/shared/models/yolo/yolov8s.onnx"
    #classes: path/to/classes.file
    output_type: yolov8
    height: 640
    width: 640
    detection_options:
      confidence: 0.33
      nms:
        enabled: yes
        threshold: 0.44
```
