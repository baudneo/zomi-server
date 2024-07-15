# `substitutions` section
>[!TIP]
> Sub vars use a bash like syntax: `${VAR_NAME}`

The [`substitutions`](../../configs/example_server.yml?plain=1#L5) section is where you define your substitution variables. These variables can be used throughout the
config file for convenience.

## `IncludeFile`
- `IncludeFile: <string>`
- Default: `${CFG_DIR}/secrets.yml`

The [`IncludeFile`](../../configs/example_server.yml?plain=1#L20) key is used to import additional sub vars 
from a separate file. This is useful for keeping sensitive information out of the main config file (secrets). 
If this is defined and the file does not exist, the server will fail to run.

>[!TIP]
> `IncludeFile` can be omitted if you do not have a secrets file.

## Example
### `secrets.yml`
```yaml 
server:
  IMPORTED SECRET: "This is from the secrets file!"
```
### `server.yml`
```yaml
substitutions:
  EXAMPLE: "World!"
  BASE_DIR: /opt/zomi/server
  CFG_DIR: ${BASE_DIR}/conf
  LOG_DIR: ${BASE_DIR}/logs
  
  # - Import additional sub vars from this file
  IncludeFile: /path/to/secrets.yml

Example of a sub var: "Hello, ${EXAMPLE}"
Example of a secret: my secret = ${IMPORTED SECRET}
```