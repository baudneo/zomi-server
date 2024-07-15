# `opencv` model config
The OpenCV model config is used for models that are supported by OpenCV. Currently, DarkNet is supported 
and so is basic logic for ONNX models. Future work will include all `sub_framework` types achieving parity with 
the DarkNet `sub_framework`.

>[!WARNING]
> There is basic ONNX support due to < 4.8.x having ONNX model issues. 
> There were open issues, which may be resolved by now.

## `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `opencv` `framework` models.

## `config`
- `config: <string:path>`
- Default: None

The `config` key is used to set the path to the config file for models that require it. 

>[!WARNING]
> `framework`:`opencv` `*.weight` models require a `config` file.

## `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!TIP]
> If the `classes` key is omitted, the COCO 2017 classes (80 labels) are used.

## `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* [`height`](../../../configs/example_server.yml?plain=1#L164) and 
[`width`](../../../configs/example_server.yml?plain=1#L165) keys are used to set the image dimensions 
to resize to before passing to the model.

## `framework`
- `framework: opencv`

## `sub_framework`
- `sub_framework: <string>`
- `darknet` or `onnx`
- *WIP* `caffe`, `trt`, `torch`, `vino`, `tensorflow`

| Sub Framework | Description                       |
|---------------|-----------------------------------|
| `darknet`     | DarkNet models (YOLO version 3-7) |
| `onnx`        | ONNX models *WIP*                 |
| `caffe`       | Caffe models *WIP*                |
| `trt`         | TensorRT models *WIP*             |
| `torch`       | PyTorch models *WIP*              |
| `vino`        | OpenVINO models *WIP*             |
| `tensorflow`  | TensorFlow models *WIP*           |

## `square`
- `square: <string>` 
- `yes` or `no`
- Default: `no`

The `square` key is used to set whether to square the image by zero-padding the shorter side to 
match the longer side before resize (AKA letterboxing).

## `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device: 
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

## `cuda_fp_16`
- `cuda_fp_16: <string>`
- `yes` or `no`
- Default: `no`

The `cuda_fp_16` key is used to enable or disable FP16 inference on Nvidia GPUs for this model.

## `output_type`
>[!WARNING]
> **This is a WIP feature and is not implemented yet.**

- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server 
how to process the output from the model into confidence, bounding box, and class id.

## Example
```yaml
models:
  # OpenCV DarkNet Model Example
  - name: YOLOv4
    description: An OpenCV DarkNet YOLOv4 model example
    enabled: no
    framework: opencv
    sub_framework: darknet
    processor: cpu
#    gpu_idx: 1
#    cuda_fp_16: no
#    output_type: yolov4  # WIP
    input: "${MODEL_DIR}/yolo/yolov4_new.weights"
    config: "${MODEL_DIR}/yolo/yolov4_new.cfg"
    #classes: "${MODEL_DIR}/coco.names"  # When omitted, COCO 17 (80) classes are used
    height: 512
    width: 512
    square: no
    detection_options:
      confidence: 0.2
      nms: 0.4
```
