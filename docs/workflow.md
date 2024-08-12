# How does this work?

## The link between zomi-server and zomi-client
>[!IMPORTANT]
> The server defines models in its configuration file [`models:`](Config/models.md) section. The `name:` of the model is how it will 
> be called in inference requests. The client will use the [`name:`](Config/models.md#name) to send requests to the server.

The `name:` of each of the `models:` defined is the link between zomi-server and zomi-client.

### Example
- A client sends an inference request to the server with at least 1 image and 1 model name; `yolo v10`.
- The server will look in its internal state to see if there is a model named `yolo v10`.
- If the model is found, enabled and no issues loading into memory, the server will run the image through the model and return the results.
    - if the model is not found/enabled/loaded, the server will return an error message. *WIP* 

## Configuration (server.yml)
The server is configured using a YAML file. Any changes require a server restart to take effect. 
The server will not start if the config file is missing or malformed.

The [example_server.yml](../configs/example_server.yml) file is used as a template for the `install.py` script. 

There are [documents available](Config/README.md) for each section of the config file.

### Models
Models are defined in the config file [`models:`](../configs/example_server.yml?plain=1#L127) section. 
Model names should be unique and are assigned a UUID on startup.

See the [models docs](Config/models.md) for more info.

### Substitution Variables 

>[!TIP]
> Sub vars use a bash like syntax: `${VAR_NAME}`
> 
> See the [substitutions docs](Config/substitutions.md) for more info.

Substitution variables (sub vars) are used in the config file for convenience and to keep sensitive information 
out of the main config file for when you are sharing it with others (secrets).

### Secrets (secrets.yml)
Secrets are sub vars that are stored in a separate file. The secrets file is not required for the server to start 
if it is not defined in the config file [`substitutions:IncludeFile:`](../configs/example_server.yml?plain=1#L20).

## Requests
>[!IMPORTANT]
> All requests require a valid JWT token. If you haven't enabled auth in the `server.yml` config file, any username:password combo will work.

- Someone logs in and receives a JWT token, the token is used in the header of all requests
- A request is made to the server, regardless of if auth is enabled or not, the server will check the validity of the token

### Inference
- an inference request is made using JSON:
    ```json
    {
        "images": ["base64 encoded image #1", "base64 encoded image #2"],
        "model_hints": "model name/UUID #1, model name/UUID #2"
    }
    ```
- the server will decode the images and run them through the models, if the models exist and are enabled
- *[Optional]* - the server will run color detection on the cropped bounding boxes
- the server will return the results in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema)

### Annotate Images
- an annotation request is made using JSON:
    ```json
    {
        "images": ["base64 encoded image #1", "base64 encoded image #2"],
        "model_hints": "model name/UUID #1, model name/UUID #2"
    }
    ```
- the server will decode the images and run them through the models, if the models exist and are enabled
- the server will return the annotated images and the results in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema)

### Available Models
>[!NOTE]
> There are several different available model endpoints based on the model type, processor or framework and one that shows all models.

- the server will return a list of available models in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema and available endpoints)