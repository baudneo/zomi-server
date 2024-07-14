# How does this work?

## Configuration (server.yml)
The server is configured using a YAML file. Any changes require a server restart to take effect. 
The server will not start if the config file is missing or malformed.

The [example_server.yml](../configs/example_server.yml) file is used as a template for the `install.py` script. 
You can also read the comments in the file for more information until proper documentation is written.

### Models
Models are defined in the config file [`models:`](../configs/example_server.yml?plain=1#L127) section. Model names should be unique and are assigned a UUID on startup.

### Substitution Variables 

>[!TIP]
> Sub vars use a bash like syntax: `${VAR_NAME}`

Substitution variables (sub vars) are used in the config file for convenience and to keep sensitive information 
out of the main config file for when you are sharing it with others (secrets).

### Secrets (secrets.yml)
Secrets are sub vars that are stored in a separate file. The secrets file is not required for the server to start 
if it is not defined in the config file [`substitutions:IncludeFile:`](../configs/example_server.yml?plain=1#L20).

### Example
#### `secrets.yml`
```yaml 
server:
  IMPORTED SECRET: "This is from the secrets file!"
```
#### `server.yml`
```yaml
substitutions:
  EXAMPLE: "World!"
  BASE_DIR: /opt/zomi/server
  CFG_DIR: ${BASE_DIR}/conf
  LOG_DIR: ${BASE_DIR}/logs
  
  # Import additional sub vars from this file
  IncludeFile: /path/to/secrets.yml

Example of a sub var: "Hello, ${EXAMPLE}"
Example of a secret: my secret = ${IMPORTED SECRET}
```

## Requests
>[!IMPORTANT]
> All requests require a valid JWT token. If you haven't enabled auth in the `server.yml` config file, any username:password combo will work.

- Someone logs in and receives a JWT token, the token is used in the header of all requests

### Inference
- A request is made to the server, regardless of if auth is enabled or not, the server will check the validity of the token
- an inference request is made using JSON:
    ```json
    {
        "images": ["base64 encoded image #1", "base64 encoded image #2"],
        "model_hints": "model name/UUID #1, model name/UUID #2"
    }
    ```
- the server will decode the images and run them through the models
- *[Optional]* - the server will run color detection on the cropped bounding boxes
- the server will return the results in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema)

### Annotate Images
- A request is made to the server, regardless of if auth is enabled or not, the server will check the validity of the token
- an annotation request is made using JSON:
    ```json
    {
        "images": ["base64 encoded image #1", "base64 encoded image #2"],
        "model_hints": "model name/UUID #1, model name/UUID #2"
    }
    ```
- the server will decode the images and run them through the models
- the server will return the annotated images and the results in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema)

### Available Models
>[!NOTE]
> There are several different available model endpoints based on the model type, processor or framework and one that shows all models.

- A request is made to the server, regardless of if auth is enabled or not, the server will check the validity of the token
- the server will return a list of available models in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema and available endpoints)