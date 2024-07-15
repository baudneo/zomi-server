# `torch` model config
There is basic pretrained model support for PyTorch models. User supplied models are not supported at this time.

## `pretrained` subsection
The `pretrained` subsection is where you define the torch pretrained model settings.

>[!NOTE]
> Only a pretrained or user supplied model are supported. If `pretrained` is enabled, 
> `input`, `classes` and `num_classes` (the 3 required keys for a user supplied model) are ignored.

> [!IMPORTANT]
> When using a pretrained model, the `TORCH_HOME` environment variable must be set to the path where the model weights are stored.
> There is logic to change it to the configured model dir so multiple copies of the models are not created.
> The `TORCH_HOME` variable is reset to whatever it was before the server ran inference.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The `enabled` key is used to enable or disable the pretrained model. It's either a pretrained or users 
model defined by `input`, `classes` and `num_classes`. 

### `name`
- `name: <string>`
- `default` or `balanced` or `accurate` or `fast` or `high_performance` or `low_performance` (*WIP*)
- Default: `default` / `balanced`
The `name` key is used to set the pretrained model from included torch models:

#### Named pretrained models
| Name                    | Description                                                                                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `accurate`              | Slower but more accurate -> [**fRCNN MN v3**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)                                |
| `fast`                  | Faster but less accurate -> [**FCOS RN50 v2**](https://pytorch.org/vision/2.0/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn) |
| `default` or `balanced` | Balanced (Default) -> [**RetinaNet RN50 v2**](https://pytorch.org/vision/master/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)                                      |
| `high_performance`      | High performance settings -> [**fRCNN RN50 v2**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)                                    |
| `low_performance`       | Low performance settings -> *WIP* [**SSDlite ?**](https://pytorch.org/vision/main/models/ssdlite.html)                                                                                             |

## `num_classes`
- `num_classes: <int>`

The `num_classes` key is used to set the number of classes including background. This is only for torch (`.pt`) models.

## `gpu_idx`
- `gpu_idx: <int>`

The `gpu_idx` key is used to set the index of the GPU to use. 

> [!NOTE]
> The index is zero based, so the first GPU is 0, second is 1, etc.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device: 
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

## `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings. 

>[!CAUTION]
> Some `framework` models use a different nms format.

#### `enabled`
- `enabled: yes` or `no`

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
  - name: TORch tESt  # lower-cased, spaces preserved = torch test
    description: testing pretrained torch model example
    enabled: no
    framework: torch
    pretrained:
      enabled: yes
      name: default
    #input: /path/to/model.pt  # WIP / NOT IMPLEMENTED
    #classes: /path/to/classes/file.txt  # WIP / NOT IMPLEMENTED
    #num_classes: 80  # WIP / NOT IMPLEMENTED
    model_type: object
    processor: gpu
    gpu_idx: 0
    detection_options:
      confidence: 0.2
      nms: 
        enabled: yes
        threshold: 0.332
```