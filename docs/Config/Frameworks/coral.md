# `coral` (TPU) model config
The `coral` `framework` is specifically for the [coral.ai Edge TPU](https://coral.ai/products/). 
Only the USB version has been tested with zomi-server.

>[!IMPORTANT]
> The pre-built google `pycoral` library only supports Python 3.7-3.9
> There is a 3.10 prebuilt available from a user on GitHub, but it is not official.
> You can also build `pycoral` yourself for python 3.11+.
> See [this issue thread](https://github.com/google-coral/pycoral/issues/85) for 
> 3.10 prebuilt repo and build instructions (including required hacks)

## `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `coral` `framework` models.

## `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!CAUTION]
> All the included pretrained TPU object detection models require the 90 label COCO dataset.
> You can find the list of labels [here](See https://coral.ai/models/object-detection/)

## `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

## `processor`
- `processor: <string>`
- Default: `tpu`

The `processor` key is used to set the processor to use for that model. It will always be set to `tpu`

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

## Example
```yaml
models:
  - name: tpu
    description: "SSD MobileNet V2 TensorFlow2 trained"
    enabled: no
    framework: coral
    type_of: object
    processor: tpu
    input: "${MODEL_DIR}/coral_tpu/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    classes: "${MODEL_DIR}/coral_tpu/coco-labels-paper.txt"
    #height: 512  ## Neither are required but can be supplied.
    #width: 512
    detection_options:
      confidence: 0.2
      nms:
        enabled: yes
        threshold: .35
```