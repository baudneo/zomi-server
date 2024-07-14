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
The `substitutions` section is where you define your substitution variables. These variables can be used throughout the
config file for convenience.

### `IncludeFile`
- `IncludeFile: /path/to/secrets.yml`

The `IncludeFile` key is used to import additional sub vars from a separate file. This is useful for keeping sensitive
information out of the main config file (secrets). If this is defined and the file does not exist, the server will fail to run.


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
The `uvicorn` section is where you define the Uvicorn server settings. These settings are passed directly to Uvicorn
and are used to configure the underlying ASGI server.

### `proxy_headers`
- `proxy_headers: yes` or `no`

The `proxy_headers` key is used to configure Uvicorn to trust the headers from a proxy server. This is useful when
running behind a reverse proxy server like Nginx or Apache.

### `forwarded_allow_ips`
- `- ip.address`

The `forwarded_allow_ips` key is used to configure Uvicorn to trust the `X-Forwarded-For` header from a proxy server.

## `debug`
- `debug: yes` or `no`

The `debug` key is used to enable or disable debug mode. This is useful for troubleshooting issues with the 
underlying ASGI server.

### Example
```yaml
uvicorn:
  proxy_headers: yes
  forwarded_allow_ips:
    - 10.0.1.1 
  debug: no
```

## `system` section
The `system` section is where you define system settings.

### `config_path`
- `config_path: /path/to/config/dir`

The `config_path` key is used to define the path where the zomi-server will store configuration files.

### `variable_data_path`
- `variable_data_path: /path/to/variable/data/dir`

The `variable_data_path` key is used to define the path where the zomi-server will 
store variable data (tokens, serialized data, etc).

### `tmp_path`
- `tmp_path: /path/to/tmp/dir`

The `tmp_path` key is used to define the path where temp files will be stored.

### `image_dir`
- `image_dir: /path/to/image/dir`

The `image_dir` key is used to define the path where various images will be stored.

### `model_dir`
- `model_dir: /path/to/model/dir`

The `model_dir` key is used to define the path where the ML model folder structure will be stored.

### `thread_workers`
- `thread_workers: <number of max parallel processes>`

The `thread_workers` key is used to define the maximum threaded processes. Adjust this to your core count and load.

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
The `server` section is where you define the server settings. There is an `auth` subsection where you can enable or disable
authentication and set the authentication settings.

### `address`
- `address: ip.address`

The `address` key is used to set the interface IP to listen on.

### `port`
- `port: <port_num>`

The `port` key is used to set the port to listen on.

### `auth` subsection

#### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable authentication. 
If disabled, anyone can access the API (any username:password combo accpeted) but, they must still 
login and receive a token.

#### `db_file` (REQUIRED)
- `db_file: /path/to/user/db`

The `db_file` key is used to set where to store the user database
>[!IMPORTANT]
> The `db_file` key is **required**

#### `sign_key` (REQUIRED)
- `sign_key: <JWT_SIGN_PHRASE>`

The `sign_key` key is used to set the JWT signing key
>[!IMPORTANT]
> The `sign_key` key is **required**

#### `algorithm`
- `algorithm: HS256`

The `algorithm` key is used to set the JWT signing algorithm

#### `expire_after`
- `expire_after: <time_in_minutes>`

The `expire_after` key is used to set the JWT token expiration time in minutes

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
> Locks use asyncio.BoundedSemaphore. No file locking.

The `locks` section is where you define the lock settings.

### `gpu` subsection
The `gpu` subsection is where you define the GPU lock settings.

#### `max`
- `max: <number of max parallel processes>`

The `max` key is used to define the maximum parallel inference requests running on the GPU.

### `cpu` subsection
The `cpu` subsection is where you define the CPU lock settings.

#### `max`
- `max: <number of max parallel processes>`

The `max` key is used to define the maximum parallel inference requests running on the CPU.

### `tpu` subsection
The `tpu` subsection is where you define the TPU lock settings.

#### `max`
- `max: <number of max parallel processes>`

The `max` key is used to define the maximum parallel inference requests running on the TPU.
>[!CAUTION]
> For TPU, unexpected results may occur when max > 1, **YMMV**.

### Example
```yaml
locks:
  # - Locks have been changed to use asyncio.BoundedSemaphore. No more file locking.
  gpu:
    # - Max number of parallel inference requests running on the GPU (Default: 4)
    max: 6
  cpu:
    # - Default: 4
    max: 12
  tpu:
    # - For TPU, unexpected results may occur when max > 1, YMMV.
    # - Default: 1
    max: 1
```

## `logging` section
The `logging` section is where you define the logging settings.

### `level`
- `level: debug` or `info` or `warning` or `error` or `critical`

The `level` key is used to set the **root logging level**.

### `sanitize` subsection
The `sanitize` subsection is where you define the log sanitization settings. This is useful for removing 
sensitive information from logs like tokens, keys, passwords, usernames, host and ip addresses.

#### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable log sanitization.

#### `replacement_str`
- `replacement_str: string_to_replace_sensitive_info`

The `replacement_str` key is used to set the string that will replace the sensitive information.

### `console` subsection
The `console` subsection is where you define the console (stdout) logging settings.

#### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable console logging.

#### `level`
- `level: debug` or `info` or `warning` or `error` or `critical`

Different log types can have different logging levels. This is where you define the console logging level 
**if you want it to be different than the root logging level**.

### `syslog` subsection
The `syslog` subsection is where you define the syslog logging settings.

#### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable syslog logging.

#### `address`
- `address: /dev/log` or `ip.address`

The `address` key is used to set the syslog address.

#### `level`
- `level: debug` or `info` or `warning` or `error` or `critical`

Different log types can have different logging levels. This is where you define the syslog logging level
**if you want it to be different than the root logging level**.

### `file` subsection
The `file` subsection is where you define the file logging settings.

#### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable file logging.

#### `level`
- `level: debug` or `info` or `warning` or `error` or `critical`

Different log types can have different logging levels. This is where you define the file logging level

#### `path`
- `path: /path/to/log/dir`

The `path` key is used to set the directory where log files will be stored.

#### `file_name`
- `file_name: log_file_name.log`

The `file_name` key is used to set the name of the log file.

#### `user` and `group`
- `user: log_file_owner` and `group: log_file_group`

The `user` and `group` keys are used to override the log file owner and group.

### Example
```yaml
logging:
  # - Logging levels are: debug, info, warning, error, critical
  # - Root logging level
  level: ${ML_INSTALL_LOGGING_LEVEL}

  # - Try to sanitize tokens, keys, passwords, usernames, host and ip addresses from logs
  sanitize:
    enabled: yes
    replacement_str: <sanitized>
  # - Log to console aka stdout
  console:
    enabled: ${ML_INSTALL_LOGGING_CONSOLE_ENABLED}
    # - Per module logging level
    #level: debug
  syslog:
    enabled: ${ML_INSTALL_LOGGING_SYSLOG_ENABLED}
    #level: debug
    address: ${ML_INSTALL_LOGGING_SYSLOG_ADDRESS}
  # - Log to file
  file:
    enabled: ${ML_INSTALL_LOGGING_FILE_ENABLED}
    #level: debug
    # - Directory where log files will be stored
    path: ${LOGGING_DIR}
    # - File name for the log file
    file_name: zomi_server.log
    # - Override log file owner and group
    # user:
    # group:
```

## `models` section

The `models` section is a list of defined model settings. Each model config is a dictionary.

### `models > name`
- `name: model_name`

The `name` key is used to set the model name. This is used when sending an inference request.
>[!TIP]
> the name is lower-cased and preserves spaces. `YOLO v10` will be lower-cased to `yolo v10`, 
> `TorcH TesT` will be lower-cased to `torch test`.

### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable the model.

### `description`
- `description: model description: can be short and sweet or detailed.`

The `description` key is used to set the model description. This key and value are not used for anything other than
documentation.

### `type_of`
- `type_of: object` or `face` or `alpr`

The `type_of` key is used to set the type of model. This is used to determine how to process the output from the model.

>[!NOTE]
> The `type_of` key can change what model keys are available! Different combinations of `type_of`, 
> `framework` and `sub-framework` can result in different model keys being available.

### `framework`
- `framework: opencv` or `trt` or `ort` or `torch` or `coral` or `http` or `face_recognition` or `alpr`

The `framework` key is used to set the ML framework to use.

### `sub_framework`
- `sub_framework: darknet` or `onnx` or `platerecognizer` or `rekognition`

The `sub_framework` key is used to set the sub-framework to use. The `sub_framework` changes 
based on the `framework` key.

### `processor`
- `processor: cpu` or `gpu` or `tpu` or `none`

The `processor` key is used to set the processor to use for that model.

>[!TIP]
> When using `framework: http`, the `processor` key is ignored/will always be `none`..

### `input`
- `input: /path/to/model/file`

The `input` key is used to set the path to the model file for most `framework` types 
and is **REQUIRED** for those models.

### `config`
- `config: /path/to/config/file`

The `config` key is used to set the path to the config file for models that require it. 
So far, only `.weights` files require a `.cfg` file.

### `classes`
- `classes: /path/to/classes/file`

The `classes` key is used to set the path to the classes file for the model. 
If this is not configured, the default COC17 (80) classes will be used.

### `height` and `width`
- `height: <int>` and `width: <int>`

The `height` and `width` keys are used to set the image dimensions to resize to before passing to the model.
Both default to `416`.

### `square`
- `square: yes` or `no`

The `square` key is used to set whether to square the image by zero-padding the shorter side to 
match the longer side before resize (AKA letterboxing).

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.
Things like confidence thresholds, NMS thresholds, etc.

#### `confidence`
- `confidence: <float:0.01-1.0>`

The `confidence` key is used to set the confidence threshold for detection. I recommend 0.2-0.5 to keep 
the noise down but also allow the client to do some filtering

#### `nms`
- `nms: <float:0.01-1.0>`

The `nms` key is used to set the Non-Max Suppressive threshold. Lower will filter more overlapping bounding boxes out.

>[!IMPORTANT]
> torch, ort, trt and coral `framework` models use a different nms format.
```yaml
nms:
    enabled: yes # no
    threshold: 0.4
```

# Models
Models are defined in the config file `models:` section. Model names should be unique and are assigned a UUID on startup.

The above breakdown/example in the [configuration file > models](#models-section) section shows keys 
that can be used across all model types. This section will show the specific keys for each model type that can be added 
or clarification of allowed values.

>[!IMPORTANT]
> Different `framework`, `type_of` and `sub-framework` values will result in different model keys being available.

## Pytorch model config
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
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable the pretrained model. It's either a pretrained or users 
model defined by `input`, `classes` and `num_classes`. 

#### `name`
- `name: default` or `balanced` or `accurate` or `fast` or `high_performance` or `low_performance` (*WIP*)

The `name` key is used to set the pretrained model from included torch models:
- `accurate` - Slower but more accurate -> [**fRCNN MN v3**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html)
- `fast` - Faster but less accurate -> [**FCOS RN50 v2**](https://pytorch.org/vision/2.0/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn)
- `default` or `balanced` - Balanced (Default) -> [**RetinaNet RN50 v2**](https://pytorch.org/vision/master/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)
- `high_performance` - High performance settings -> [**fRCNN RN50 v2**](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
- `low_performance` - Low performance settings -> *WIP* [**SSDlite ?**](https://pytorch.org/vision/main/models/ssdlite.html)

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
> To print out the number of available GPUs: `python3 -c "import torch; print(torch.cuda.device_count())"`
> To get the name of each device: `python3 -c "import torch; 
> print(' '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))"`

### `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

#### `nms` subsection
The `nms` subsection is where you define the Non-Max Suppressive settings. 
Some `framework` models use a different nms format.

##### `enabled`
- `enabled: yes` or `no`

The `enabled` key is used to enable or disable Non-Max Suppressive filtering.

##### `threshold`
- `threshold: <float:0.01-1.0>`

The `threshold` key is used to set the Non-Max Suppressive threshold. Lower will filter more overlapping bounding boxes out.

### Example
```yaml
models:
  # PyTorch Example (very rudimentary, basics plumbed in)
  - name: TORch tESt  # lower-cased, spaces preserved = torch test
    description: testing pretrained torch model
    enabled: no
    framework: torch
    pretrained:
      enabled: yes
      name: default
    #input: /path/to/model.pt
    #classes: /path/to/classes/file.txt
    #num_classes: 80
    type_of: object
    processor: gpu
    # - If using multiple GPUs, set the index of the GPU to use, Ignored if processor is not gpu
    # - To print out number of available GPUs: python3 -c "import torch; print(torch.cuda.device_count())"
    # - To get the name of each device: python3 -c "import torch; print(' '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))"
    # ** NOTE: The index is zero based, so the first GPU is 0, second is 1, etc.
    gpu_idx: 0

    detection_options:
      confidence: 0.2
      nms: 
        enabled: yes
        threshold: 0.65
```

## OpenCV model config
The OpenCV model config is used for models that are supported by OpenCV. Currently, DarkNet is supported 
and so is basic logic for ONNX models.

>[!WARNING]
> OpenCV ONNX has basic support due to 4.8.x having ONNX model issues. 
> There was an open issue which may be resolved by now. 
> The `darknet`:`framework` `onnx`:`sub-framework` will be worked on in the future.

### `config`
- `config: /path/to/config/file`

The `config` key is used to set the path to the config file for models that require it 
(usually only .weights files require a .cfg file).

### `framework`
- `framework: opencv`

Set the ML framework to use `opencv`.

### `sub_framework`
- `sub_framework: darknet` or `onnx`

The `sub_framework` key is used to set the sub-framework to use. The `sub_framework` changes
based on the `framework` key.

- `darknet` - DarkNet models (YOLO v 3, 4, 7)
- `onnx` - ONNX models *WIP*

### Example
```yaml
models:
  # OpenCV DarkNet Model Example
  - name: YOLOv4 
    enabled: no
    framework: opencv
    sub_framework: darknet
    processor: cpu
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

### Example
```yaml
models:
  # - TPU Model Example [pre-built google pycoral only supports Python 3.7-3.9]
  # - There is a 3.10 prebuilt available from a user on GitHub, but it is not official.
  # - You can also build it yourself for python 3.11+.
  # - See https://github.com/google-coral/pycoral/issues/85 for 3.10 prebuilt repo and build instructions (including required hacks)
  - name: tpu
    description: "SSD MobileNet V2 TensorFlow2 trained"
    enabled: no

    framework: coral
    type_of: object
    # - Will always be tpu
    processor: tpu

    input: "${MODEL_DIR}/coral_tpu/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    # - All of the included TPU object detection models require the 90 label COCO dataset
    # - See https://coral.ai/models/object-detection/
    classes: "${MODEL_DIR}/coral_tpu/coco-labels-paper.txt"

    detection_options:
      # - Non Max Suppressive threshold, lower will filter more overlapping bounding boxes out.
      # - Currently, only TPU model NMS can be enabled/disabled
      nms:
        enabled: yes
        threshold: .35
      confidence: 0.2

  # TensorRT Model Example (User must install TensorRT and compile their engine model)
  - name: yolo-nas-s trt
    enabled: no
    description: "TensorRT optimized YOLO-NAS-S pretrained model"
    input: "/shared/models/yolo/yolo_nas_s.trt"
    #gpu_idx: 0
    framework: trt

    type_of: object
    # - Only ort and trt support output_type
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

  # ONNX Runtime Example
  - name: yolov8s onnx
    description: "Ultralytics YOLO v8s pretrained ONNX model on onnxruntime"
    enabled: no
    # - Use onnxruntime backend: ort
    framework: ort

    type_of: object

    processor: gpu
#    gpu_idx: 0
    # - Possible ROCm support *WIP*
#    gpu_framework: cuda

    input: "/shared/models/yolo/yolov8s.onnx"
    #classes: path/to/classes.file

    # - Only ort and trt support output_type
    # - Tells the server how to process the output from the model into confidence, bounding box, and class id
    # - [yolonas, yolov8, yolov10]
    output_type: yolov8

    # ** NOTE: MAKE SURE to set the proper model input size or it will throw errors!
    height: 640
    width: 640

    detection_options:
      confidence: 0.33
      # ort and trt support nms enabled/disabled (YOLO v10 does not need NMS)
      nms:
        enabled: yes
        threshold: 0.44

  # AWS Rekognition Example *WIP*
  - name: aws
    description: "AWS Rekognition remote HTTP detection (PAID per request!)"
    enabled: no
    framework: http
    sub_framework: rekognition
    type_of: object
    processor: none

    detection_options:
      confidence: 0.4455

  # face-recognition Example
  - name: dlib face
    enabled: no
    description: "dlib face detection/recognition model"
    type_of: face
    framework: face_recognition

    # - These options only apply to when the model is
    # - used for training faces to be recognized
    training_options:
      # - 'cnn' is more accurate but slower on CPUs. 'hog' is faster but less accurate
      # ** NOTE: if you use cnn here you MUST use cnn for detection
      model: cnn
      # - How many times to upsample the image looking for faces.
      # - Higher numbers find smaller faces but take longer.
      upsample_times: 1
      # - How many times to re-sample the face when calculating encoding.
      # - Higher is more accurate, but slower (i.e. 100 is 100x slower)
      num_jitters: 1
      # - Max width of image to feed the model (scaling applied)
      max_size: 600
      # - Source dir where known faces are stored.
      dir: "${DATA_DIR}/face_data/known"

    # - An unknown face is a detected face that does not match any known faces
    # - Can possibly use the unknown face to train a new face depending on the quality of the cropped image
    unknown_faces:
      enabled: yes
      # - The label to give an unknown face
      label_as: "Unknown"
      # - When cropping an unknown face from the frame, how many pixels to add to each side
      leeway_pixels: 10
      # - Where to save unknown faces
      dir: "${DATA_DIR}/face_data/unknown"

    detection_options:
      # - 'cnn' is more accurate but slower on CPUs. 'hog' is faster but less accurate
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

  # - OpenALPR local binary Example
  - name: "openalpr cpu"
    # ** NOTE:
    # - You need to understand how to skew and warp images to make a plate readable by OCR, see openalpr docs.
    # - You can also customize openalpr config files and only run them on certain cameras.
    description: "openalpr local SDK (binary) model with a config file for CPU"
    enabled: no

    type_of: alpr
    framework: alpr
    processor: none
    # - ALPR sub-framework to use: openalpr, platerecognizer
    sub_framework: openalpr

    detection_options:
      binary_path: alpr
      # - The default config file uses CPU, no need to make a custom config file
      binary_params:
      confidence: 0.5
      max_size: 600

  # - OpenALPR Model Example (Shows how to use a custom openalpr config file)
  - name: "openalpr gpu"
    description: "openalpr local SDK (binary) with a config file to use CUDA GPU"
    enabled: no

    type_of: alpr
    framework: alpr
    sub_framework: openalpr
    # - openalpr config file controls processor, you can put none,cpu,gpu or tpu here.
    processor: none

    detection_options:
      # - Path to alpr binary (default: alpr)
      binary_path: alpr
      # - Make a config file that uses the gpu instead of cpu
      binary_params: "--config /etc/alpr/openalpr-gpu.conf"
      confidence: 0.5
      max_size: 600

  # - Plate Recognizer Example
  - name: 'Platerec'
    enabled: no
    type_of: alpr
    # - Even though it is ALPR, it is using HTTP for detection
    framework: http
    sub_framework: plate_recognizer

    api_type: cloud
    #api_url: "https://api.platerecognizer.com/v1/plate-reader/"
    api_key: ${PLATEREC_API_KEY}
    detection_options:
      # - Only look in certain countrys or regions in a country.
      # - See platerecognizer docs for more info
      #regions: ['ca', 'us']

      stats: no

      min_dscore: 0.5

      min_score: 0.5

      max_size: 1600

      # - For advanced users, you can pass in any of the options from the API docs
      #payload:
        #regions: ['us']
        #camera_id: 12

      #config:
        #region: 'strict'
        #mode:  'fast'
```