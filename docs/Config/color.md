# `color` section
The [`color`](../../configs/example_server.yml?plain=1#L417) section is where you define the color detection settings.

>[!IMPORTANT]
> In the model config, there is a boolean key `detect_color` that can override the global 
> `color_detection>enabled` key *for that specific monitor*.

## `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L420) key is used to enable or disable color detection.

## `top_n`
- `top_n: <int>`
- Default: `5`

The [`top_n`](../../configs/example_server.yml?plain=1#L422) key is used to set the number of top colors to return.

## `spec`
- `spec: <string>`
- `html4` or `css2` or `css21` or `css3`
- Default: `html4`

The [`spec`](../../configs/example_server.yml?plain=1#L424) key is used to set the color specification to use. 

>[!IMPORTANT]
> The `spec` will change how the color string is returned. html might = `gray`, css3 might = `lightgray`, etc.

## `labels` subsection
The [`labels`](../../configs/example_server.yml?plain=1#L426) subsection is where you define the labels that 
color detection should be run on.

>[!TIP]
> If no labels are configured, color detection will run on all detected objects.

### Entry format
>[!CAUTION]
> **This is a list entry**
- `- <string>`

## Example
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