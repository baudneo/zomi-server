# Rekognition
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

# Plate Recognizer Snapshot cloud API

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
