# `alpr` model config
ALPR models have `sub_framework` and then another subtype `api_type` that can be `cloud` or `local`.

## Example
### `openalpr` local examples
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

#### `framework`
- `framework: <string>`
- `alpr`

The `framework` key is used to set the ML framework to use.

#### `sub_framework`
- `sub_framework: <string>`
- `openalpr`

The `sub_framework` key is used to set the sub-framework to use.

#### `processor`
- `processor: <string>`
- `none`

The `processor` key is used to set the processor to use for that model. It will always be set to `none`
when `sub_framework` is `openalpr`. This is because it is an external binary.

#### `detection_options` subsection

##### `max_size`
- `max_size: <int>`
- Default: None

The `max_size` key is used to set the max width of the image to feed the model (scaling applied).

##### `binary_path`
- `binary_path: <string:path>`
- Default: `alpr`

The `binary_path` key is used to set the path to the OpenALPR binary.

##### `binary_params`
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