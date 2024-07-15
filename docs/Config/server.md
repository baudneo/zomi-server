# `server` section
The [`server`](../../configs/example_server.yml?plain=1#L56) section is where you define the server settings. There is an 
[`auth`](../../configs/example_server.yml?plain=1#L60) subsection where you can enable or disable authentication and set the authentication settings.

## `address`
- `address: <string:ip address>`
- Default: `0.0.0.0`

The [`address`](../../configs/example_server.yml?plain=1#L58) key is used to set the interface IP to listen on.

## `port`
- `port: <int>`
- Default: `5000`

The [`port`](../../configs/example_server.yml?plain=1#L59) key is used to set the port to listen on.

## `auth` subsection
The [`auth`](../../configs/example_server.yml?plain=1#L60) subsection is where you define the authentication settings.

### `enabled`
- `enabled: <string>` 
- `yes` or `no`
- Default: `no`

The [`enabled`](../../configs/example_server.yml?plain=1#L64) key is used to enable or disable authentication. 

>[!IMPORTANT]
> If **disabled**, _anyone can access the API_ **(any username:password combo accepted)** but, they must still 
> login, receive a token and use that token in every request.

### `db_file`
- `db_file: <string:path>` **REQUIRED**
- Default: `${DATA_DIR}/udata.db`

The [`db_file`](../../configs/example_server.yml?plain=1#L67) key is used to set where to store the user database
>[!IMPORTANT]
> The `db_file` key is **required**

### `sign_key`
- `sign_key: <string>` **REQUIRED**
- Default: None

The [`sign_key`](../../configs/example_server.yml?plain=1#L71) key is used to set the JWT signing key
>[!IMPORTANT]
> The `sign_key` key is **required**

### `algorithm`
- `algorithm: <string>`
- Default: `HS256`

The [`algorithm`](../../configs/example_server.yml?plain=1#L74) key is used to set the JWT signing algorithm
#### Algorithm Values
| Algorithm Value  | Digital Signature or MAC Algorithm  |
|------------------|-------------------------------------|
| HS256	           | HMAC using SHA-256 hash algorithm   |
| HS384	           | HMAC using SHA-384 hash algorithm   |
| HS512	           | HMAC using SHA-512 hash algorithm   |
| RS256	           | RSASSA using SHA-256 hash algorithm |
| RS384	           | RSASSA using SHA-384 hash algorithm |
| RS512	           | RSASSA using SHA-512 hash algorithm |
| ES256	           | ECDSA using SHA-256 hash algorithm  |
| ES384	           | ECDSA using SHA-384 hash algorithm  |
| ES512	           | ECDSA using SHA-512 hash algorithm  |

### `expire_after`
- `expire_after: <int>`
- Default: `60`

The [`expire_after`](../../configs/example_server.yml?plain=1#L77) key is used to set the JWT token 
expiration time in minutes

## Example
```yaml
server:
  address: ${SERVER_ADDRESS}
  port: ${SERVER_PORT}
  auth:
    enabled: no
    db_file: ${DATA_DIR}/udata.db
    sign_key: ${JWT_SIGN_PHRASE}
    algorithm: HS256
    expire_after: 60
```