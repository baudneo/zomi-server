# Configuration File
The YAML configuration file allows using **Substitution Variables** for convenience and **Secrets** to keep sensitive 
information out of the main config file.

## Substitution Variables
>[!TIP]
> Sub vars use a bash like syntax: `${VAR_NAME}`

See the [substitutions docs](substitutions.md) for more info.

Substitution variables (sub vars) are used in the config file for convenience and to keep sensitive information 
out of the main config file for when you are sharing it with others (secrets).

### Secrets (secrets.yml)
Secrets are sub vars that are stored in a separate file. The secrets file is not required for the server to start 
if it is not defined in the config file [`substitutions:IncludeFile:`](../../configs/example_server.yml?plain=1#L20).

## `uvicorn` section
See the [uvicorn docs](uvicorn.md) for more info.

## `system` section
See the [system docs](system.md) for more info.

## `logging` section
See the [logging docs](logging.md) for more info.

## `color` section
See the [color docs](color.md) for more info.

## `locks` section
See the [locks docs](locks.md) for more info.

## `models` section
See the [models docs](models.md) for more info.

### `framework`
There are other specific model config options available based on the `framework` used.

See the [framework docs](Frameworks/README.md) for more info.