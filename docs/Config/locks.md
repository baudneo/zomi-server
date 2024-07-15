# `locks` section
>[!NOTE]
> Locks use asyncio.BoundedSemaphore.

The [`locks`](../../configs/example_server.yml?plain=1#L79) section is where you define the processor lock settings.

## `gpu` subsection
The [`gpu`](../../configs/example_server.yml?plain=1#L81) subsection is where you define the GPU lock settings.

### `max`
- `max: <int>`
- Default: `4`

The [`max`](../../configs/example_server.yml?plain=1#L83) key is used to define the maximum parallel inference requests running on the GPU.

## `cpu` subsection
The [`cpu`](../../configs/example_server.yml?plain=1#L84) subsection is where you define the CPU lock settings.

### `max`
- `max: <int>` 
- Default: `4`

The [`max](../configs/example_server.yml?plain=1#L86)` key is used to define the maximum parallel inference requests running on the CPU.

## `tpu` subsection
The [`tpu`](../../configs/example_server.yml?plain=1#L87) subsection is where you define the TPU lock settings.

### `max`
- `max: <int>`
- Default: `1`

The [`max`](../../configs/example_server.yml?plain=1#L90) key is used to define the maximum parallel inference requests running on the TPU.
>[!CAUTION]
> For TPU, unexpected results may occur when max > 1, **YMMV**.

## Example
```yaml
locks:
  gpu:
    max: 6
  cpu:
    max: 12
  tpu:
    max: 1
```