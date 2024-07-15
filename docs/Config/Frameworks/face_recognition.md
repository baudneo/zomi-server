# `face_recognition` model config
The `face_recognition` `framework` is used for face detection and recognition based on D-Lib.

## `framework`
- `framework: <string>`
- `face_recognition`

The `framework` key is used to set the ML framework to use.

## `training_options` subsection
The `training_options` subsection is where you define the training settings for the model.
Training is done via cli, you supply <x> 'passport' style photos of a person to train a recognition model.

When a face is detected, it is compared to the known faces and if it matches, 
the label is set to the known face label (the name of the file, i.e. `john.1.jpg`).

### `model`
- `model: <string>`
- `cnn` or `hog`
- Default: `cnn`

The `model` key is used to set the model to use for training. `cnn` is far more accurate but slower on CPUs.
`hog` is faster but less accurate.

>[!IMPORTANT]
> If you use `cnn` for training, you MUST use `cnn` for detection.

### `upsample_times`
- `upsample_times: <int>`
- Default: `1`

The `upsample_times` key is used to set how many times to upsample the image looking for faces.
Higher numbers find smaller faces but will take longer.

### `num_jitters`
- `num_jitters: <int>`
- Default: `1`

The `num_jitters` key is used to set how many times to re-sample the face when calculating encoding.
Higher is more accurate but slower (i.e. 100 is 100x slower).

### `max_size`
- `max_size: <int>`
- Default: `600`

The `max_size` key is used to set the max width of the image to feed the model (scaling applied).
The image will be resized to this width before being passed to the model.

>[!NOTE]
> The larger the max size, the longer it will take and the more memory it will use.
> If you see out of memory errors, lower the max size.

### `dir`
- `dir: <string:path>`
- Default: None

The `dir` key is used to set the source directory where known (trained) faces are stored.

## `unknown_faces` subsection
The `unknown_faces` subsection is where you define the settings for unknown faces.
An unknown face is a detected face that does not match any known faces. The goal of unknown face cropping is to
get a good image of the unknown face to possibly train a new face.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `yes`

The `enabled` key is used to enable or disable the unknown face cropping.

### `label_as`
- `label_as: <string>`
- Default: `Unknown`

The `label_as` key is used to set the label to give an unknown face.

### `leeway_pixels`
- `leeway_pixels: <int>`
- Default: `10`

The `leeway_pixels` key is used to set how many pixels to add to each side when cropping 
an unknown face from the image.

### `dir`
- `dir: <string:path>`
- Default: None

The `dir` key is used to set the directory where unknown faces are stored.

## `detection_options` subsection
The `detection_options` subsection is where you define the detection settings for the model.

### `model`
- `model: <string>`
- `cnn` or `hog`
- Default: `cnn`

The `model` key is used to set the model to use for detection. `cnn` is far more accurate but slower on CPUs.
`hog` is faster but less accurate.

>[!IMPORTANT]
> If you use `cnn` for training, you MUST use `cnn` for detection.

### `confidence`
- `confidence: <float>`
- Range: `0.01 - 1.0`
- Default: `0.2`

The `confidence` key is used to set the confidence threshold for detection of a face.

### `recognition_threshold`
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
      model: cnn
      confidence: 0.5
      recognition_threshold: 0.6
      upsample_times: 1
      num_jitters: 1
      max_size: 600
```