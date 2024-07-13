# ZoneMinder Machine Learning API (zomi-server)
:warning: This software is in **ALPHA** stage, expect issues. :warning:

This is a FastAPI based server component for the new ZoneMinder ML add-on. It processes HTTP JSON requests with what models to run 
on BASE64 encoded images and returns the results. The client is responsible for filtering the detection results. 
There is also rudimentary support for color detection of detected objects bounding boxes.

The server currently supports several ML backends and hardware accelerators:

**ML backends:**
- OpenCV (DarkNet)
- PyTorch
- ONNXRuntime (YOLO v8, YOLO-NAS, [YOLO v10 *WIP*])
- TensorRT (YOLO v8, YOLO-NAS, [YOLO v10 *WIP*])
- pycoral (coral.ai Edge TPU)
- openalpr (local binary)
- [face-recognition](https://github.com/ageitgey/face_recognition) based on D-Lib
- HTTP
    - platerecognizer.com
    - AWS Rekognition

**Hardware**
- CPU
- Nvidia GPU
- AMD GPU (Pytorch/onnxruntime ROCm) *WIP*
- Coral.ai Edge TPU


# Install

The [install.py](examples/install.py) file creates a venv for the project and installs the required packages. 
Pytorch and onnxruntime are installed by default, face-recognition with D-Lib and opencv are optional (`--face-recognition` and `--opencv`).

## Requirements
- System packages (package names from debian based system)
    - `python3-venv`
    - `python3-pip`
    - `python3-setuptools`
    - `python3-wheel`
- Python 3.8+ with packages
    - `requests`
    - `psutil`
    - `distro`
    - `tqdm`
- `--user` and `--group` arguments are required
- `--cpu` or `--gpu` argument is required

>[!CAUTION]
> the pypi (pip) based `pycoral` libs only support python 3.7-3.9. If you are using python 3.10, you will need 
> to install the `pycoral` libs manually, please see: [this issue](https://github.com/google-coral/pycoral/issues/85) and read through the whole comment thread 

### Processor type
>[!IMPORTANT]
> The `--cpu` **or** `--gpu <choice>` argument is required. 

`--gpu` choices:
- `cuda12.1`
- `cuda11.8`
- `rocm6.0`

>[!NOTE]
> The `--cpu` argument does not require a value, it is a flag to install the CPU based packages.

### Install as user
>[!IMPORTANT]
> The `--user` and `--group` arguments are required. 

These are the user and group that the server will run as.

### Examples
```bash
python3 examples/install.py --cpu --debug --dry-run --user <username> --group <groupname>
```

```bash
python3 examples/install.py --gpu cuda12.1 --debug --dry-run --user <username> --group <groupname>
```

>[!NOTE]
> `--dry-run` will show the commands that will be run without actually running them.

# Swagger UI
The server has a built-in Swagger UI for testing the API. It is available at the server root: `http://<server>:<port>/`

>![IMPORTANT]
> All requests require a valid JWT token. If you havent turned on auth OR created any users. Any username:password combo will work.

# User authentication
>[!NOTE]
> You can enable and disable authentication, but all requests must have a valid JWT token. When authentication is disabled,
> the login endpoint will accept any username:password combo and supply a valid usable token.

The server uses a JWT system for authentication. When a new user is created, the default user will be 
automatically disabled. If all users are deleted, the server will enable the default user again.

The current user database is based on the python `tinydb` module.

## Default user
The default user is `imoz` with the password `zomi`. Checks are performed that ensure the default user is created on startup.

# User Management
User management is done using the CLI `mlapi` script and the `user` sub-command.

```bash
mlapi user --help
```

# Start the server
The server can be started with the `mlapi` script.

```bash
mlapi -C /path/to/config/file.yml
```
## Debugging
The server can be started in debug mode with the `--debug` or `-D` flag.

```bash
mlapi -C /path/to/config/file.yml --debug
```