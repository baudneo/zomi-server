# `uvicorn` section
The [`uvicorn`](../../configs/example_server.yml?plain=1#L29) section is where you define the Uvicorn server settings. 
These settings are passed directly to Uvicorn and are used to configure the underlying ASGI server.

## `proxy_headers`
- `proxy_headers:<string>` 
- `yes` or `no`
- Default: `no`

The [`proxy_headers`](../../configs/example_server.yml?plain=1#L32) key is used to configure Uvicorn to trust 
the headers from a proxy server. This is useful when running behind a reverse proxy server like Nginx or Apache.

## `forwarded_allow_ips` subsection
- Default: None

The [`forwarded_allow_ips`](../../configs/example_server.yml?plain=1#L37) subsection is where you define the IP 
addresses that Uvicorn will trust the `X-Forwarded-For` header from. This is useful when running behind a 
reverse proxy server like Nginx or Apache.

### Entry format
- `- <string:ip address>`
>[!CAUTION]
> **This is a list entry**
> 
> CIDR notation is not supported, only single IP addresses are allowed. This is a 
> limitation of the underlying libraries.

## `debug`
- `debug: <string>`
- `yes` or `no`
- Default: `no`

The [`debug`](../../configs/example_server.yml?plain=1#L40) key is used to enable or disable debug mode. This is useful for troubleshooting issues with the 
underlying ASGI server.

## Example
```yaml
uvicorn:
  proxy_headers: yes
  forwarded_allow_ips:
    - 10.0.1.1
    - 12.34.56.78
  debug: no
```