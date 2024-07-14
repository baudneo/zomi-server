# Configuration File
The YAML configuration file allows using **Substitution Variables** for convenience and **Secrets** to keep sensitive 
information out of the main config file.

## Substitution Variables
>[!TIP]
> Sub vars use a bash like syntax: `${VAR_NAME}`

Substitution variables (sub vars) are used in the config file for convenience and to keep sensitive information 
out of the main config file for when you are sharing it with others (secrets).

## Secrets (secrets.yml)
Secrets are sub vars that are stored in a separate file. The secrets file is not required for the server to start 
if it is not defined in the config file [`substitutions:IncludeFile:`](../configs/example_server.yml?plain=1#L20).


## `substitutions` section
The [`substitutions`](../configs/example_server.yml?plain=1#L5) section is where you define your substitution variables. These variables can be used throughout the
config file for convenience.

### `IncludeFile`
- `IncludeFile: <string>`
- Default: `${CFG_DIR}/secrets.yml`

The [`IncludeFile`](../configs/example_server.yml?plain=1#L20) key is used to import additional sub vars 
from a separate file. This is useful for keeping sensitive information out of the main config file (secrets). 
If this is defined and the file does not exist, the server will fail to run.


### Example
#### `secrets.yml`
```yaml 
server:
  IMPORTED SECRET: "This is from the secrets file!"
```
#### `server.yml`
```yaml
substitutions:
  EXAMPLE: "World!"
  BASE_DIR: /opt/zomi/server
  CFG_DIR: ${BASE_DIR}/conf
  LOG_DIR: ${BASE_DIR}/logs
  
  # Import additional sub vars from this file
  IncludeFile: /path/to/secrets.yml

Example of a sub var: "Hello, ${EXAMPLE}"
Example of a secret: my secret = ${IMPORTED SECRET}
```

## `uvicorn` section
The [`uvicorn`](../configs/example_server.yml?plain=1#L29) section is where you define the Uvicorn server settings. 
These settings are passed directly to Uvicorn and are used to configure the underlying ASGI server.

### `proxy_headers`
- `proxy_headers:<string>` 
- `yes` or `no`
- Default: `no`

The [`proxy_headers`](../configs/example_server.yml?plain=1#L32) key is used to configure Uvicorn to trust 
the headers from a proxy server. This is useful when running behind a reverse proxy server like Nginx or Apache.

### `forwarded_allow_ips` subsection
- Default: None

The [`forwarded_allow_ips`](../configs/example_server.yml?plain=1#L37) subsection is where you define the IP 
addresses that Uvicorn will trust the `X-Forwarded-For` header from. This is useful when running behind a 
reverse proxy server like Nginx or Apache.

#### Entry format
- `- <string:ip address>`
>[!CAUTION]
> **This is a list entry**
> 
> CIDR notation is not supported, only single IP addresses are allowed. This is a 
> limitation of the underlying libraries.

## `debug`
- `debug: <string>`
- `yes` or `no`
- Default: `no`

The [`debug`](../configs/example_server.yml?plain=1#L40) key is used to enable or disable debug mode. This is useful for troubleshooting issues with the 
underlying ASGI server.

### Example
```yaml
uvicorn:
  proxy_headers: yes
  forwarded_allow_ips:
    - 10.0.1.1
    - 12.34.56.78
  debug: no
```

## `system` section
The [`system`](../configs/example_server.yml?plain=1#L42) section is where you define system settings.

### `config_path`
- `config_path: <string:path>`
- Default: `${BASE_DIR}/conf`

The [`config_path`](../configs/example_server.yml?plain=1#L44) key is used to define the path where zomi-server will store configuration files.

### `variable_data_path`
- `variable_data_path: <string:path>`
- Default: `${BASE_DIR}/data`

The [`variable_data_path`](../configs/example_server.yml?plain=1#L46) key is used to define the path where zomi-server will 
store variable data (tokens, serialized data, etc).

### `tmp_path`
- `tmp_path: <string:path>`
- Default: `${BASE_DIR}/tmp`

The [`tmp_path`](../configs/example_server.yml?plain=1#L48) key is used to define the path where zomi-server will store
temp files.

### `image_dir`
- `image_dir: <string:path>`
- Default: `${DATA_DIR}/images`

The [`image_dir`](../configs/example_server.yml?plain=1#L50) key is used to define the path where 
various images will be stored.

### `model_dir`
- `model_dir: <string:path>`
- Default: `${BASE_DIR}/models`

The [`model_dir`](../configs/example_server.yml?plain=1#L52) key is used to define the path where the ML model 
folder structure will be stored.

### `thread_workers`
- `thread_workers: <int>`
- Default: `4`

The [`thread_workers`](../configs/example_server.yml?plain=1#L54) key is used to define the maximum 
threaded processes. Adjust this to your core count and load.

### Example
```yaml
system:
  config_path: ${CFG_DIR}
  variable_data_path: ${DATA_DIR}
  tmp_path: ${TMP_DIR}
  image_dir: ${DATA_DIR}/images
  model_dir: ${MODEL_DIR}
  thread_workers: 4
  ```

## `server` section
The [`server`](../configs/example_server.yml?plain=1#L56) section is where you define the server settings. There is an 
[`auth`](../configs/example_server.yml?plain=1#L60) subsection where you can enable or disable authentication and set the authentication settings.

### `address`
- `address: <string:ip address>`
- Default: `0.0.0.0`

The [`address`](../configs/example_server.yml?plain=1#L58) key is used to set the interface IP to listen on.

### `port`
- `port: <int>`
- Default: `5000`

The [`port`](../configs/example_server.yml?plain=1#L59) key is used to set the port to listen on.

### `auth` subsection
The [`auth`](../configs/example_server.yml?plain=1#L60) subsection is where you define the authentication settings.

#### `enabled`
- `enabled: <string>` 
- `yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L64) key is used to enable or disable authentication. 

>[!IMPORTANT]
> If **disabled**, _anyone can access the API_ **(any username:password combo accepted)** but, they must still 
> login, receive a token and use that token in every request.

#### `db_file`
- `db_file: <string:path>` **REQUIRED**
- Default: `${DATA_DIR}/udata.db`

The [`db_file`](../configs/example_server.yml?plain=1#L67) key is used to set where to store the user database
>[!IMPORTANT]
> The `db_file` key is **required**

#### `sign_key`
- `sign_key: <string>` **REQUIRED**
- Default: None

The [`sign_key`](../configs/example_server.yml?plain=1#L71) key is used to set the JWT signing key
>[!IMPORTANT]
> The `sign_key` key is **required**

#### `algorithm`
- `algorithm: <string>`
- Default: `HS256`

The [`algorithm`](../configs/example_server.yml?plain=1#L74) key is used to set the JWT signing algorithm
##### Algorithm Values
| Algorithm Value  | Digital Signature or MAC Algorithm  |
|------------------|-------------------------------------|
| HS256	           | HMAC using SHA-256 hash algorithm   |
| HS384	           | HMAC using SHA-384 hash algorithm   |
| HS512	           | HMAC using SHA-512 hash algorithm   |
| RS256	           | RSASSA using SHA-256 hash algorithm |
| RS384	           | RSASSA using SHA-384 hash algorithm |
| RS512	           | RSASSA using SHA-512 hash algorithm |
| ES256	           | ECDSA using SHA-256 hash algorithm  |
| ES384	           | ECDSA using SHA-384 hash algorithm  |
| ES512	           | ECDSA using SHA-512 hash algorithm  |

#### `expire_after`
- `expire_after: <int>`
- Default: `60`

The [`expire_after`](../configs/example_server.yml?plain=1#L77) key is used to set the JWT token 
expiration time in minutes

### Example
```yaml
server:
  address: ${SERVER_ADDRESS}
  port: ${SERVER_PORT}
  auth:
    enabled: no
    db_file: ${DATA_DIR}/udata.db
    sign_key: ${JWT_SIGN_PHRASE}
    algorithm: HS256
    expire_after: 60
```

## `locks` section
>[!NOTE]
> Locks use asyncio.BoundedSemaphore.

The [`locks`](../configs/example_server.yml?plain=1#L79) section is where you define the processor lock settings.

### `gpu` subsection
The [`gpu`](../configs/example_server.yml?plain=1#L81) subsection is where you define the GPU lock settings.

#### `max`
- `max: <int>`
- Default: `4`

The [`max`](../configs/example_server.yml?plain=1#L83) key is used to define the maximum parallel inference requests running on the GPU.

### `cpu` subsection
The [`cpu`](../configs/example_server.yml?plain=1#L84) subsection is where you define the CPU lock settings.

#### `max`
- `max: <int>` 
- Default: `4`

The [`max](../configs/example_server.yml?plain=1#L86)` key is used to define the maximum parallel inference requests running on the CPU.

### `tpu` subsection
The [`tpu`](../configs/example_server.yml?plain=1#L87) subsection is where you define the TPU lock settings.

#### `max`
- `max: <int>`
- Default: `1`

The [`max`](../configs/example_server.yml?plain=1#L90) key is used to define the maximum parallel inference requests running on the TPU.
>[!CAUTION]
> For TPU, unexpected results may occur when max > 1, **YMMV**.

### Example
```yaml
locks:
  gpu:
    max: 6
  cpu:
    max: 12
  tpu:
    max: 1
```

## `logging` section
The [`logging`](../configs/example_server.yml?plain=1#L92) section is where you define the logging settings.

### `level`
- `level: <string>` 
- `debug` or `info` or `warning` or `error` or `critical`
- Default: `info`

The [`level`](../configs/example_server.yml?plain=1#L95) key is used to set the **root logging level**.

### `sanitize` subsection
The [`sanitize`](../configs/example_server.yml?plain=1#L98) subsection is where you define the log sanitization settings. This is useful for removing 
sensitive information from logs like tokens, keys, passwords, usernames, host and ip addresses.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L99) key is used to enable or disable log sanitization.

#### `replacement_str`
- `replacement_str: <string>`
- Default: `<sanitized>`

The [`replacement_str`](../configs/example_server.yml?plain=1#L100) key is used to set the string that will replace the sensitive information.

### `console` subsection
The [`console`](../configs/example_server.yml?plain=1#L102) subsection is where you define the console (stdout) logging settings.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L103) key is used to enable or disable console logging.

#### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../configs/example_server.yml?plain=1#L105) key is used to set the console logging level.

>[!TIP]
> Different log types can have different logging levels. 
> **if you want it to be different from the root logging level**.

### `syslog` subsection
The [`syslog`](../configs/example_server.yml?plain=1#L106) subsection is where you define the syslog logging settings.

#### `enabled`
- `enabled: yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L107) key is used to enable or disable syslog logging.

#### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../configs/example_server.yml?plain=1#L108) key is used to set the syslog logging level.

#### `address`
- `address: <string>`
- Default: `/dev/log`

The [`address`](../configs/example_server.yml?plain=1#L109) key is used to set the syslog address.

### `file` subsection
The [`file`](../configs/example_server.yml?plain=1#L111) subsection is where you define the file logging settings.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L112) key is used to enable or disable file logging.

#### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../configs/example_server.yml?plain=1#L113) key is used to set the file logging level.2

#### `path`
- `path: <string>`
- Default: `${LOG_DIR}`

The [`path`](../configs/example_server.yml?plain=1#L115) key is used to set the directory where log files will be stored.

#### `file_name`
- `file_name: <string>`
- Default: `zomi_server.log`

The [`file_name`](../configs/example_server.yml?plain=1#L117) key is used to set the name of the log file.

#### `user` and `group`
- `user: <string>` and `group: <string>`
- Default: None

The [`user`](../configs/example_server.yml?plain=1#L119) and [`group`](../configs/example_server.yml?plain=1#L120) 
keys are used to override the log file owner and group.

### Example
```yaml
logging:
  level: info

  sanitize:
    enabled: yes
    replacement_str: <nuh-uh>
  console:
    enabled: yes
    level: debug
  syslog:
    enabled: no
    level: warning
    address: 10.0.1.34
  file:
    enabled: yes
    level: warning
    path: ${LOGGING_DIR}
    file_name: zomi_server.log
    user: log-user
    group: log-group
```

## `models` section

The [`models`](../configs/example_server.yml?plain=1#L140) section is a list of defined model settings.
>[!IMPORTANT]
> This section only covers the base model config settings (all models will have these settings).
> Each model `framework` has its own specific settings that are not covered here. For more information on
> specific model settings, see the [Models](#models) section.

>[!CAUTION]
> **Each model config is a list entry stored as a dictionary.**

### models > `name`
- `name: <string>` **REQUIRED**
- Default: None
  
The [`name`](../configs/example_server.yml?plain=1#L142) key is used to set the model name. The `name`
is used when sending an inference request.
>[!IMPORTANT]
> The `name` key is **REQUIRED** and must be unique. The name is lower-cased and preserves spaces.
> `YOLO v10` will be lower-cased to `yolo v10`, `TorcH TesT` will be lower-cased to `torch test`.

### models > `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The [`enabled`](../configs/example_server.yml?plain=1#L143) key is used to enable or disable the model.

### models > `description`
- `description: <string>`
- Default: None

The [`description`](../configs/example_server.yml?plain=1#L144) key is used to set the model description. 
This key and value are not used for anything other than documentation.

### models > `type_of`
- `type_of: <string>` or `model_type: <string>`
- `object` or `face` or `alpr`
- Default: `object`

The [`type_of`](../configs/example_server.yml?plain=1#L145) key is used to set the type of model. 
This is used by the zomi-client to determine how to filter the output from the model.

>[!NOTE]
> The `type_of` key can change what model keys are available! Different combinations of `type_of`, 
> `framework` and `sub_framework` can result in different model keys being available.

### `framework`
- `framework: <string>` 
- `opencv` or `trt` or `ort` or `torch` or `coral` or `http` or `face_recognition` or `alpr`
- Default: `ort`

The [`framework`](../configs/example_server.yml?plain=1#L148) key is used to set the ML framework to use.

### `sub_framework`
- `sub_framework: <string>`
- See the [table below](#available-sub-framework-values) for available sub-frameworks
- Default: `darknet`

The [`sub_framework`](../configs/example_server.yml?plain=1#L151) key is used to set the sub-framework to use.
>[!IMPORTANT]
> The `sub_framework` choices change based on the `framework` key. 
> This key can be omitted, some models don't process this key.

#### Available `sub-framework` values
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

### `processor`
- `processor: <string>` 
- `cpu` or `gpu` or `tpu` or `none`
- Default: `cpu`

The [`processor`](../configs/example_server.yml?plain=1#L154) key is used to set the processor to use for that model.

>[!TIP]
> When using `framework`:`http`, the `processor` key is ignored/will always be `none`.
> When using `framework`:`coral`, the `processor` key is ignored/will always be `tpu`.

### `detection_options` subsection
The [`detection_options`](../configs/example_server.yml?plain=1#L163) subsection is where you define the 
detection settings for the model. Things like confidence thresholds, NMS thresholds, etc.

#### `confidence`
- `confidence: <float>`
- Range: `0.01 - 1.0`
- Default: `0.2`

The [`confidence`](../configs/example_server.yml?plain=1#L165) key is used to set the confidence 
threshold for detection. I recommend 0.2-0.5 to keep the noise down but also allow the client to do some filtering

### `detect_color`
- `detect_color: <string>`
- `yes` or no`
- Default: `no`

The [`detect_color`](../configs/example_server.yml?plain=1#L161) key is used to ovveride the global color detection
`enabled` flag. [`color_detection`](#color-section) is configured globally.

## `color` section
The [`color`](../configs/example_server.yml?plain=1#L417) section is where you define the color detection settings.

>[!IMPORTANT]
> In the model config, there is a boolean key `detect_color` that can override the global 
> `color_detection>enabled` key *for that specific monitor*.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../configs/example_server.yml?plain=1#L420) key is used to enable or disable color detection.

### `top_n`
- `top_n: <int>`
- Default: `5`

The [`top_n`](../configs/example_server.yml?plain=1#L422) key is used to set the number of top colors to return.

### `spec`
- `spec: <string>`
- `html4` or `css2` or `css21` or `css3`
- Default: `html4`

The [`spec`](../configs/example_server.yml?plain=1#L424) key is used to set the color specification to use. 

>[!IMPORTANT]
> The `spec` will change how the color string is returned. html might = `gray`, css3 might = `lightgray`, etc.

### `labels` subsection
The [`labels`](../configs/example_server.yml?plain=1#L426) subsection is where you define the labels that 
color detection should be run on.

>[!TIP]
> If no labels are configured, color detection will run on all detected objects.

#### Entry format
>[!CAUTION]
> **This is a list entry**
- `- <string>`
### Example
```yaml
color:
  enabled: no
  top_n: 4
  spec: html4
  labels:
    - car
    - truck
    - bus
    - motorcycle
    - bicycle
```

# Models / Frameworks
Models are defined in the config file `models:` section. Model names should be unique and are assigned a UUID on startup.

The above breakdown/example in the [configuration file > models](#models-section) section shows keys 
that can be used across all model types. This section will show the specific keys for each model type that can be added 
or clarification of allowed values.

>[!IMPORTANT]
> Different `framework`, `type_of` and `sub_framework` values will result in different model keys being available.

## `torch` model config
There is basic pretrained model support for PyTorch models. User supplied models are not supported at this time.

### `pretrained` subsection
The `pretrained` subsection is where you define the torch pretrained model settings.

>[!NOTE]
> Only a pretrained or user supplied model are supported. If `pretrained` is enabled, 
> `input`, `classes` and `num_classes` (the 3 required keys for a user supplied model) are ignored.

> [!IMPORTANT]
> When using a pretrained model, the `TORCH_HOME` environment variable must be set to the path where the model weights are stored.
> There is logic to change it to the configured model dir so multiple copies of the models are not created.
> The `TORCH_HOME` variable is reset to whatever it was before the server ran inference.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The `enabled` key is used to enable or disable the pretrained model. It's either a pretrained or users 
model defined by `input`, `classes` and `num_classes`. 

#### `name`
- `name: <string>`
- `default` or `balanced` or `accurate` or `fast` or `high_performance` or `low_performance` (*WIP*)
- Default: `default` / `balanced`
The `name` key is used to set the pretrained model from included torch models:

##### Named pretrained models
| Name                    | Description                                                                                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `accurate`              | Slower but more accurate -> [**fRCNN MN v3**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)                                |
| `fast`                  | Faster but less accurate -> [**FCOS RN50 v2**](https://pytorch.org/vision/2.0/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn) |
| `default` or `balanced` | Balanced (Default) -> [**RetinaNet RN50 v2**](https://pytorch.org/vision/master/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)                                      |
| `high_performance`      | High performance settings -> [**fRCNN RN50 v2**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)                                    |
| `low_performance`       | Low performance settings -> *WIP* [**SSDlite ?**](https://pytorch.org/vision/main/models/ssdlite.html)                                                                                             |

### `num_classes`
- `num_classes: <int>`

The `num_classes` key is used to set the number of classes including background. This is only for torch (`.pt`) models.

### `gpu_idx`
- `gpu_idx: <int>`

The `gpu_idx` key is used to set the index of the GPU to use. 

> [!NOTE]
> The index is zero based, so the first GPU is 0, second is 1, etc.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device: 
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings. 

>[!CAUTION]
> Some `framework` models use a different nms format.

##### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

##### `threshold`
- `threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.35`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more 
overlapping bounding boxes out.

### Example
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

## `opencv` model config
The OpenCV model config is used for models that are supported by OpenCV. Currently, DarkNet is supported 
and so is basic logic for ONNX models. Future work will include all `sub_framework` types achieving parity with 
the DarkNet `sub_framework`.

>[!WARNING]
> There is basic ONNX support due to < 4.8.x having ONNX model issues. 
> There were open issues, which may be resolved by now.

### `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `opencv` `framework` models.

### `config`
- `config: <string:path>`
- Default: None

The `config` key is used to set the path to the config file for models that require it. 

>[!WARNING]
> `framework`:`opencv` `*.weight` models require a `config` file.

### `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!TIP]
> If the `classes` key is omitted, the COCO 2017 classes (80 labels) are used.

### `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* [`height`](../configs/example_server.yml?plain=1#L164) and 
[`width`](../configs/example_server.yml?plain=1#L165) keys are used to set the image dimensions 
to resize to before passing to the model.

### `framework`
- `framework: opencv`

### `sub_framework`
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

### `square`
- `square: <string>` 
- `yes` or `no`
- Default: `no`

The `square` key is used to set whether to square the image by zero-padding the shorter side to 
match the longer side before resize (AKA letterboxing).

### `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device: 
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

### `cuda_fp_16`
- `cuda_fp_16: <string>`
- `yes` or `no`
- Default: `no`

The `cuda_fp_16` key is used to enable or disable FP16 inference on Nvidia GPUs for this model.

### `output_type`
>[!WARNING]
> **This is a WIP feature and is not implemented yet.**

- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server 
how to process the output from the model into confidence, bounding box, and class id.

### Example
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

## `ort` (ONNXRuntime) model config
The `ort` `framework` is used for ONNX models that are supported by ONNXRuntime.

### `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device:
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

### `gpu_framework`
>[!WARNING]
> **This is a WIP feature and is not implemented yet.**

- `gpu_framework: <string>`
- `cuda` or `rocm` *WIP*

The `gpu_framework` key is used to set the GPU framework to use. Hopefully we get ROCm support for 
torch and onnxruntime (`ort`)

### `output_type`
- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server
how to process the output from the model into confidence, bounding box, and class id.

### `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings.

##### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

##### `threshold`
- `threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.35`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more
overlapping bounding boxes out.

### Example
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

## `coral` (TPU) model config
The `coral` `framework` is specifically for the coral.ai Edge TPU. 
Only the USB version has been tested with zomi-server.

>[!IMPORTANT]
> The pre-built google `pycoral` library only supports Python 3.7-3.9
> There is a 3.10 prebuilt available from a user on GitHub, but it is not official.
> You can also build `pycoral` yourself for python 3.11+.
> See [this issue thread](https://github.com/google-coral/pycoral/issues/85) for 
> 3.10 prebuilt repo and build instructions (including required hacks)

### `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `coral` `framework` models.

### `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!CAUTION]
> All the included pretrained TPU object detection models require the 90 label COCO dataset.
> You can find the list of labels [here](See https://coral.ai/models/object-detection/)

### `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

### `processor`
- `processor: <string>`
- Default: `tpu`

The `processor` key is used to set the processor to use for that model. It will always be set to `tpu`

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings.

##### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

##### `threshold`
- `threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.35`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more

### Example
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

## `trt` (Tensor RT) model config

### `input`
- `input: <string:path>` **REQUIRED**
- Default: None

The `input` key is used to set the path to the model file

>![!IMPORTANT]
> The `input` key is **REQUIRED** for `trt` `framework` models.

### `classes`
- `classes: <string:path>`
- Default: COC0 2017 classes (80 labels)

The `classes` key is used to set the path to the classes file for the model.

>[!TIP]
> If the `classes` key is omitted, the COCO 2017 classes (80 labels) are used.

### `height` and `width`
- `height: <int>` and `width: <int>`
- Default: `416`

The *input* `height` and `width` keys are used to set the image dimensions

### `processor`
- `processor: <string>`
- Default: `gpu`

The `processor` key is used to set the processor to use for that model. It will always be set to `gpu`

### `gpu_idx`
- `gpu_idx: <int>`
- Default: 0

The `gpu_idx` key is used to set the index of the GPU to use.

>[!TIP]
> If using multiple GPUs, set the index of the GPU to use. Ignored if `processor` is not `gpu`.
> To get the index and name of each device:
> `python3 -c 'import torch; print(f"Available GPUs: {torch.cuda.device_count()}") [print(f"index: {i} - {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'`

### `output_type`
- `output_type: <string>`
- `yolov3` or `yolov4` or `yolov7` or `yolov8` or `yolonas` or `yolov10`
- Default: None

The `output_type` key is used to set the output type for the model. This is used to tell the server
how to process the output from the model into confidence, bounding box, and class id.

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings.

##### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

##### `threshold`
- `threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.35`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more
overlapping bounding boxes out.

### Example
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


## `http`:`rekognition` (AWS Rekognition) model config
The `http` `framework` is used for models that are supported by an HTTP API.

### `aws_access_key_id`, `aws_secret_access_key` and `region_name`
- `aws_access_key_id: <string>`, `aws_secret_access_key: <string>` and `region_name: <string>`
- Default: None

The `aws_access_key_id`, `aws_secret_access_key` and `region_name` keys are used to set the AWS credentials
for the model. Please see the [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#passing-credentials-as-parameters)
for more information on how to set these up.

### `framework`
- `framework: http`
- Default: `http`

The `framework` key is used to set the ML framework to use.

### `sub_framework`
- `sub_framework: <string>`
- `rekognition`

The `sub_framework` key is used to set the sub-framework to use.

### `processor`
- `processor: <string>`
- Default: `none`

The `processor` key is used to set the processor to use for that model. It will always be set to `none`
when http`:`rekognition` is set.

### Example
```yaml
models:
  # AWS Rekognition Example
  - name: aws
    description: "AWS Rekognition remote HTTP detection (PAID per request!)"
    enabled: no
    aws_access_key_id:
    aws_secret_access_key:
    region_name:

    framework: http
    sub_framework: rekognition
    type_of: object
    processor: none

    detection_options:
      confidence: 0.4455
```

## `face_recognition` model config
The `face_recognition` `framework` is used for face detection and recognition based on D-Lib.

### `framework`
- `framework: <string>`
- `face_recognition`

The `framework` key is used to set the ML framework to use.

### `training_options` subsection
The `training_options` subsection is where you define the training settings for the model.
Training is done via cli, you supply <x> 'passport' style photos of a person to train a recognition model.

When a face is detected, it is compared to the known faces and if it matches, 
the label is set to the known face label (the name of the file, i.e. `john.1.jpg`).

#### `model`
- `model: <string>`
- `cnn` or `hog`
- Default: `cnn`

The `model` key is used to set the model to use for training. `cnn` is far more accurate but slower on CPUs.
`hog` is faster but less accurate.

>[!IMPORTANT]
> If you use `cnn` for training, you MUST use `cnn` for detection.

#### `upsample_times`
- `upsample_times: <int>`
- Default: `1`

The `upsample_times` key is used to set how many times to upsample the image looking for faces.
Higher numbers find smaller faces but will take longer.

#### `num_jitters`
- `num_jitters: <int>`
- Default: `1`

The `num_jitters` key is used to set how many times to re-sample the face when calculating encoding.
Higher is more accurate but slower (i.e. 100 is 100x slower).

#### `max_size`
- `max_size: <int>`
- Default: `600`

The `max_size` key is used to set the max width of the image to feed the model (scaling applied).
The image will be resized to this width before being passed to the model.

>[!NOTE]
> The larger the max size, the longer it will take and the more memory it will use.
> If you see out of memory errors, lower the max size.

#### `dir`
- `dir: <string:path>`
- Default: None

The `dir` key is used to set the source directory where known (trained) faces are stored.

### `unknown_faces` subsection
The `unknown_faces` subsection is where you define the settings for unknown faces.
An unknown face is a detected face that does not match any known faces. The goal of unknown face cropping is to
get a good image of the unknown face to possibly train a new face.

#### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable the unknown face cropping.

#### `label_as`
- `label_as: <string>`
- Default: `Unknown`

The `label_as` key is used to set the label to give an unknown face.

#### `leeway_pixels`
- `leeway_pixels: <int>`
- Default: `10`

The `leeway_pixels` key is used to set how many pixels to add to each side when cropping 
an unknown face from the image.

#### `dir`
- `dir: <string:path>`
- Default: None

The `dir` key is used to set the directory where unknown faces are stored.

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `model`
- `model: <string>`
- `cnn` or `hog`
- Default: `cnn`

The `model` key is used to set the model to use for detection. `cnn` is far more accurate but slower on CPUs.
`hog` is faster but less accurate.

>[!IMPORTANT]
> If you use `cnn` for training, you MUST use `cnn` for detection.

#### `confidence`
- `confidence: <float>`
- Range: `0.01 - 1.0`
- Default: `0.2`

The `confidence` key is used to set the confidence threshold for detection of a face.

#### `recognition_threshold`
- `recognition_threshold: <float>`
- Range: `0.01 - 1.0`
- Default: `0.6`

The `recognition_threshold` key is used to set the confidence threshold for face recognition.

### Example
```yaml
models:
  - name: dlib face
    enabled: no
    description: "dlib face detection/recognition model"
    type_of: face
    framework: face_recognition
    training_options:
      model: cnn
      upsample_times: 1
      num_jitters: 1
      max_size: 600
      dir: "${DATA_DIR}/face_data/known"
    unknown_faces:
      enabled: yes
      label_as: "Unknown"
      leeway_pixels: 10
      dir: "${DATA_DIR}/face_data/unknown"
    detection_options:
      # - 'cnn' is more accurate but slower on CPUs. 'hog' is faster but less accurate
      # - If you use 'cnn' for training, you MUST use 'cnn' for detection!
      model: cnn
      # - Confidence threshold for face DETECTION
      confidence: 0.5
      # - Confidence threshold for face RECOGNITION (comparing the detected face to known faces)
      recognition_threshold: 0.6
      # - How many times to upsample the image looking for faces.
      # - Higher numbers find smaller faces but takes longer.
      upsample_times: 1
      # - How many times to re-sample the face when calculating encoding.
      # - Higher is more accurate, but slower (i.e. 100 is 100x slower)
      num_jitters: 1
      # - Max width of image to feed the model (scaling applied)
      max_size: 600
```

## `alpr` model config
ALPR models have `sub_framework` and then another subtype `api_type` that can be `cloud` or `local`.

### Example
#### `openalpr` local examples
The `openalpr` `sub_framework` is used for models that are supported by OpenALPR (local binary).

> [!IMPORTANT]
> You have to build OpenALPR from source to get the binary. There is CUDA support, 
> please see [this script](https://github.com/ShinobiCCTV/Shinobi/blob/dev/INSTALL/openalpr-gpu-easy.sh)
> for build instructions

>[!NOTE]
> You need to understand how to skew and warp images to make a plate readable by OCR, see openalpr docs.
> You can also customize openalpr config files and only run them on certain cameras.
> 
> OpenALPR is OLD software, try the `http`:`plate_recognizer` `sub_framework`, its free up to X requests.

##### `framework`
- `framework: <string>`
- `alpr`

The `framework` key is used to set the ML framework to use.

##### `sub_framework`
- `sub_framework: <string>`
- `openalpr`

The `sub_framework` key is used to set the sub-framework to use.

##### `processor`
- `processor: <string>`
- `none`

The `processor` key is used to set the processor to use for that model. It will always be set to `none`
when `sub_framework` is `openalpr`. This is because it is an external binary.

##### `detection_options` subsection

###### `max_size`
- `max_size: <int>`
- Default: None

The `max_size` key is used to set the max width of the image to feed the model (scaling applied).

###### `binary_path`
- `binary_path: <string:path>`
- Default: `alpr`

The `binary_path` key is used to set the path to the OpenALPR binary.

###### `binary_params`
- `binary_params: <string>`
- Default: None

The `binary_params` key is used to set the parameters to pass to the OpenALPR binary.

```yaml
models:
  - name: "openalpr cpu"
    description: "openalpr local SDK (binary) model with a config file for CPU"
    enabled: no

    type_of: alpr
    framework: alpr
    processor: none
    sub_framework: openalpr

    detection_options:
      confidence: 0.5
      max_size: 600
      binary_path: alpr
      # - The default config file uses CPU, no need to make a custom config file
      # - -j is already passed for JSON output
      binary_params:
      # - This shows how to pass a custom config file that would have the processor set to gpu
#      binary_params: "--config /etc/alpr/openalpr-gpu.conf"
```

## `http`:`plate_recognizer` model config
The `http` `framework` is used for models that are supported by an HTTP API.
The `plate_recognizer` `sub_framework` uses HTTP requests to the [platerecognizer.com](https://platerecognizer.com) API. 
It is free up to a certain number of requests but, requires you to create an account and create an API key.

>[!NOTE]
> Please see the [platerecognizer snapshot cloud api docs](https://guides.platerecognizer.com/docs/snapshot/api-reference#snapshot-cloud-api)
> for more information on the API and the model options.

### `type_of`
- `type_of: <string>`
- `alpr`

The `type_of` key is used to set the type of model to use.

### `framework`
- `framework: <string>`
- `http`

The `framework` key is used to set the ML framework to use.

### `sub_framework`
- `sub_framework: <string>`
- `plate_recognizer`

The `sub_framework` key is used to set the sub-framework to use.

### `api_type`
- `api_type: <string>`
- `cloud`

The `api_type` key is used to set the type of API to use. Only `cloud` is supported for `plate_recognizer`.

### `api_url`
- `api_url: <string>`
- Default: `https://api.platerecognizer.com/v1/plate-reader/`

The `api_url` key is used to set the API URL for HTTP requests to platerecognizer.com, for enterprise local API 
use or if the public API changes.

### `api_key`
- `api_key: <string>` **REQUIRED**
- Default: None

The `api_key` key is used to set the API key for HTTP requests to platerecognizer.com.

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `regions`
- `regions: ['<string>', '<string>']`
- Default: None
- Example: `regions: ['us', 'ca']`

See platerecognizer docs for more info

#### `stats`
- `stats: <string>`
- `yes` or `no`
- Default: `no`

#### `min_dscore`
- `min_dscore: <float>`
- Range: `0.01 - 1.0`
- Default: `0.5`
- Example: `min_dscore: 0.5`

Plate detection confidence score threshold.

#### `min_score`
- `min_score: <float>`
- Range: `0.01 - 1.0`
- Default: `0.5`

Plate **TEXT** recognition confidence score threshold.

#### `max_size`
- `max_size: <int>`
- Default: `1600`

The `max_size` key is used to set the max width of the image to feed the model (scaling applied).

#### `payload` subsection
The `payload` subsection is used to set the payload to pass to the API instead of the above config options.

#### `config` subsection
The `config` subsection is used to set the server-side model config. See the [platerecognizer API docs](https://guides.platerecognizer.com/docs/snapshot/api-reference#snapshot-cloud-api) for more info.

### Example
```yaml
models:
  - name: 'Platerec'  # lower-cased, spaces preserved = platerec
    enabled: no
    type_of: alpr
    framework: http
    sub_framework: plate_recognizer
    api_type: cloud
    #api_url: "https://api.platerecognizer.com/v1/plate-reader/"  # automatically set if undefined
    api_key: ${PLATEREC_API_KEY}
    detection_options:
      #regions: ['ca', 'us']
      stats: no
      min_dscore: 0.5
      min_score: 0.5
      max_size: 1600
      #payload:
        #regions: ['us']
        #camera_id: 12
      #config:
        #region: 'strict'
        #mode:  'fast'
```