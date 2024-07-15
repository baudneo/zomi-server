# `logging` section
The [`logging`](../../configs/example_server.yml?plain=1#L92) section is where you define the logging settings.

## `level`
- `level: <string>` 
- `debug` or `info` or `warning` or `error` or `critical`
- Default: `info`

The [`level`](../../configs/example_server.yml?plain=1#L95) key is used to set the **root logging level**.

## `sanitize` subsection
The [`sanitize`](../../configs/example_server.yml?plain=1#L98) subsection is where you define the log sanitization settings. This is useful for removing 
sensitive information from logs like tokens, keys, passwords, usernames, host and ip addresses.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L99) key is used to enable or disable log sanitization.

### `replacement_str`
- `replacement_str: <string>`
- Default: `<sanitized>`

The [`replacement_str`](../../configs/example_server.yml?plain=1#L100) key is used to set the string that will replace the sensitive information.

## `console` subsection
The [`console`](../../configs/example_server.yml?plain=1#L102) subsection is where you define the console (stdout) logging settings.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L103) key is used to enable or disable console logging.

### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../../configs/example_server.yml?plain=1#L105) key is used to set the console logging level.
>[!TIP]
> Different log types can have different logging levels. 
> **if you want it to be different from the root logging level**.

## `syslog` subsection
The [`syslog`](../../configs/example_server.yml?plain=1#L106) subsection is where you define the syslog logging settings.

### `enabled`
- `enabled: yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L107) key is used to enable or disable syslog logging.

### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../../configs/example_server.yml?plain=1#L108) key is used to set the syslog logging level.
>[!TIP]
> Different log types can have different logging levels. 
> **if you want it to be different from the root logging level**.

### `address`
- `address: <string>`
- Default: `/dev/log`

The [`address`](../../configs/example_server.yml?plain=1#L109) key is used to set the syslog address.

## `file` subsection
The [`file`](../../configs/example_server.yml?plain=1#L111) subsection is where you define the file logging settings.

### `enabled`
- `enabled: <string>`
- `yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L112) key is used to enable or disable file logging.

### `level`
- `level: <string>`
- `debug` or `info` or `warning` or `error` or `critical`
- Default: **root logging level**

The [`level`](../../configs/example_server.yml?plain=1#L113) key is used to set the file logging level.2
>[!TIP]
> Different log types can have different logging levels. 
> **if you want it to be different from the root logging level**.

### `path`
- `path: <string>`
- Default: `${LOG_DIR}`

The [`path`](../../configs/example_server.yml?plain=1#L115) key is used to set the directory where log files will be stored.

### `file_name`
- `file_name: <string>`
- Default: `zomi_server.log`

The [`file_name`](../../configs/example_server.yml?plain=1#L117) key is used to set the name of the log file.

### `user` and `group`
- `user: <string>` and `group: <string>`
- Default: None

The [`user`](../../configs/example_server.yml?plain=1#L119) and [`group`](../../configs/example_server.yml?plain=1#L120) 
keys are used to override the log file owner and group.

## Example
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