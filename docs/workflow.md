# How does this work?

## Configuration (server.yml)
The server is configured using a YAML file. Any changes require a server restart to take effect. 
The server will not start if the config file is missing or malformed.

The [example_server.yml](../configs/example_server.yml) file is used as a template for the `install.py` script. 
You can also read the comments in the file for more information until proper documentation is written.

## Secrets (secrets.yml)
>[!NOTE]
> Secrets are actually **Substitution Variables**. They are used in the config file 
> for convenience and can also be used as secrets!.

Secrets are stored in a separate file. This file is not required for the server to start if it is not 
defined in the config file (`substitutions:IncludeFile:`). This is a convenience feature to keep sensitive
information out of the main config file for when you need to share the config file with others.

## Requests
>[!NOTE]
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
- 
- A request is made to the server, regardless of if auth is enabled or not, the server will check the validity of the token
- the server will return a list of available models in JSON (See [Swagger UI](../README.md#swagger-ui) to view the JSON response schema and available endpoints)