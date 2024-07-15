# `system` section
The [`system`](../../configs/example_server.yml?plain=1#L42) section is where you define system settings.

## `config_path`
- `config_path: <string:path>`
- Default: `${BASE_DIR}/conf`

The [`config_path`](../../configs/example_server.yml?plain=1#L44) key is used to define the path where zomi-server will store configuration files.

## `variable_data_path`
- `variable_data_path: <string:path>`
- Default: `${BASE_DIR}/data`

The [`variable_data_path`](../../configs/example_server.yml?plain=1#L46) key is used to define the path where zomi-server will 
store variable data (tokens, serialized data, etc).

## `tmp_path`
- `tmp_path: <string:path>`
- Default: `${BASE_DIR}/tmp`

The [`tmp_path`](../../configs/example_server.yml?plain=1#L48) key is used to define the path where zomi-server will store
temp files.

## `image_dir`
- `image_dir: <string:path>`
- Default: `${DATA_DIR}/images`

The [`image_dir`](../../configs/example_server.yml?plain=1#L50) key is used to define the path where 
various images will be stored.

## `model_dir`
- `model_dir: <string:path>`
- Default: `${BASE_DIR}/models`

The [`model_dir`](../../configs/example_server.yml?plain=1#L52) key is used to define the path where the ML model 
folder structure will be stored.

## `thread_workers`
- `thread_workers: <int>`
- Default: `4`

The [`thread_workers`](../../configs/example_server.yml?plain=1#L54) key is used to define the maximum 
threaded processes. Adjust this to your core count and load.

## Example
```yaml
system:
  config_path: ${CFG_DIR}
  variable_data_path: ${DATA_DIR}
  tmp_path: ${TMP_DIR}
  image_dir: ${DATA_DIR}/images
  model_dir: ${MODEL_DIR}
  thread_workers: 4
  ```