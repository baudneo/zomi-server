# `models` section

The [`models`](../../configs/example_server.yml?plain=1#L140) section is a list of defined model settings.
>[!IMPORTANT]
> This section only covers the base model config settings (all models will have these settings).
> Each model `framework` has its own specific settings that are not covered here. For more information on
> specific model settings, see the [Models](#models) section.

>[!CAUTION]
> **Each model config is a list entry stored as a dictionary.**

## `name`
- `name: <string>` **REQUIRED**
- Default: None
  
The [`name`](../../configs/example_server.yml?plain=1#L142) key is used to set the model name. The `name`
is used when sending an inference request.
>[!IMPORTANT]
> The `name` key is **REQUIRED** and must be unique. The name is lower-cased and preserves spaces.
> `YOLO v10` will be lower-cased to `yolo v10`, `TorcH TesT` will be lower-cased to `torch test`.

## `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The [`enabled`](../../configs/example_server.yml?plain=1#L143) key is used to enable or disable the model.

## `description`
- `description: <string>`
- Default: None

The [`description`](../../configs/example_server.yml?plain=1#L144) key is used to set the model description. 
This key and value are not used for anything other than documentation.

## `type_of`
- `type_of: <string>` or `model_type: <string>`
- `object` or `face` or `alpr`
- Default: `object`

The [`type_of`](../../configs/example_server.yml?plain=1#L145) key is used to set the type of model. 
This is used by the zomi-client to determine how to filter the output from the model.

>[!NOTE]
> The `type_of` key can change what model keys are available! Different combinations of `type_of`, 
> `framework` and `sub_framework` can result in different model keys being available.

## `framework`
- `framework: <string>` 
- `opencv` or `trt` or `ort` or `torch` or `coral` or `http` or `face_recognition` or `alpr`
- Default: `ort`

The [`framework`](../../configs/example_server.yml?plain=1#L148) key is used to set the ML framework to use.

## `sub_framework`
- `sub_framework: <string>`
- See the [table below](#available-sub-framework-values) for available sub-frameworks
- Default: `darknet`

The [`sub_framework`](../../configs/example_server.yml?plain=1#L151) key is used to set the sub-framework to use.
>[!IMPORTANT]
> The `sub_framework` choices change based on the `framework` key. 
> This key can be omitted, some models don't process this key.

### Available `sub-framework` values
| Framework        | Sub-framework(s)                                                          |
|------------------|---------------------------------------------------------------------------|
| opencv           | `darknet`, `onnx` , [`caffe`, `trt`, `torch`, `vino`, `tensorflow`] *WIP* |
| torch            | None                                                                      |
| ort              | None                                                                      |
| trt              | None                                                                      |
| coral            | None                                                                      |
| http             | `none` , `rekognition`                                                    |
| face_recognition | None                                                                      |
| alpr             | `openalpr`, `plate_recognizer`, [`rekor`] *WIP*                           | 

## `processor`
- `processor: <string>` 
- `cpu` or `gpu` or `tpu` or `none`
- Default: `cpu`

The [`processor`](../../configs/example_server.yml?plain=1#L154) key is used to set the processor to use for that model.

>[!TIP]
> When using `framework`:`http`, the `processor` key is ignored/will always be `none`.
> When using `framework`:`coral`, the `processor` key is ignored/will always be `tpu`.

## `detection_options` subsection
The [`detection_options`](../../configs/example_server.yml?plain=1#L163) subsection is where you define the 
detection settings for the model. Things like confidence thresholds, NMS thresholds, etc.

### `confidence`
- `confidence: <float>`
- Range: `0.01 - 1.0`
- Default: `0.2`

The [`confidence`](../../configs/example_server.yml?plain=1#L165) key is used to set the confidence 
threshold for detection. I recommend 0.2-0.5 to keep the noise down but also allow the client to do some filtering

## `detect_color`
- `detect_color: <string>`
- `yes` or no`
- Default: `no`

The [`detect_color`](../../configs/example_server.yml?plain=1#L161) key is used to ovveride the global color detection
`enabled` flag. [`color_detection`](#color-section) is configured globally.