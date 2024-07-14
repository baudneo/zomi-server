# ZoneMinder Machine Learning API (zomi-server)
>[!CAUTION]
> :warning: This software is in **ALPHA** stage, expect issues and incomplete, unoptimized, janky code ;) :warning:

This is a FastAPI based server component for the new ZoneMinder ML add-on. It processes HTTP JSON requests with 
what models to run on BASE64 encoded images and returns the results (see [workflow](docs/workflow.md) docs for 
more info). The client is responsible for filtering the detection results. 
There is also rudimentary support for color detection of cropped bounding boxes.

The server currently has basic support for several ML backends and hardware accelerators:

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
- *Open an issue or pull request to get a backend/api supported*

**Hardware**
- CPU
    - OpenVINO is planned 
- Nvidia GPU (CUDA / cuDNN / Tensor RT)
- AMD GPU (Pytorch/onnxruntime ROCm) *WIP / Untested*
- Coral.ai Edge TPU


# Install

The [install.py](examples/install.py) file creates a venv for the project, installs the required packages and symlinks mlapi.py to /usr/local/bin/mlapi. 
Pytorch and onnxruntime are installed by default, face-recognition with D-Lib and opencv are additional (`--face-recognition` and `--opencv`).

## Requirements
- Clone this repo and cd into the repo directory.
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
> to install the `pycoral` libs manually, please see: [this issue](https://github.com/google-coral/pycoral/issues/85) 
> and read through the whole comment thread. You could also build the `pycoral` libs from source for python 3.11+ 
> using the method and hacks in that issue thread. 

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

### OpenCV
- `--opencv` installs the opencv-contrib-python package (no CUDA support)
**opencv is a requirement for zomi-server to run**.
>![TIP]
> Advanced users can compile opencv with CUDA/cuDNN support and link it into the venv to use GPU acceleration in opencv.
> A doc will be written outlining the linking process, there are several build tutorials online.

### Additional ML frameworks
- `--face-recognition` installs the [face-recognition](https://github.com/ageitgey/face_recognition) package with D-Lib (*may require building D-Lib first with CUDA for nvidia support*)
- Others are planned for the future (deepface, mmdetection, etc.)

### Examples

#### CPU
```bash
python3 examples/install.py --cpu --debug --opencv --dry-run --user <username> --group <groupname>
```
#### Nvidia GPU using CUDA 12.1
```bash
python3 examples/install.py --gpu cuda12.1 --debug --opencv --dry-run --user <username> --group <groupname>
```

#### Nvidia GPU using CUDA 11.8
```bash
python3 examples/install.py --gpu cuda11.8 --debug --opencv --dry-run --user <username> --group <groupname>
```

#### AMD GPU using ROCm 6.0
```bash
python3 examples/install.py --gpu rocm6.0 --debug --opencv --dry-run --user <username> --group <groupname>
```

>[!NOTE]
> `--dry-run` will show the commands that will be run without actually running them.

# Swagger UI
>[!TIP]
> Swagger UI is available at the server root: `http://<server>:<port>/`

The server uses FastAPIs built-in Swagger UI which shows available endpoints, response/request schema and 
serves as self-explanatory documentation.

>[!WARNING] 
> Make sure to authorize first! All requests require a valid JWT token. 
> If you haven't enabled auth in the `server.yml` config file, any username:password combo will work.
>![Authorize in Swagger UI](docs/assets/zomi-server_auth-button.png)

# User authentication
>[!CRITICAL]
> You can enable and disable authentication, but all requests must have a valid JWT token. When authentication is disabled,
> the login endpoint will accept any username:password combo and supply a valid usable token.

## Default user
The default user is `imoz` with the password `zomi`.

# User Management
User management is done using the `mlapi` script and the `user` sub-command. 
For more information, please see the [User Management](docs/user_management.md) docs.

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

# SystemD service
A SystemD service file is provided in the [configs/systemd](configs/systemd/mlapi.service) directory.

```bash
sudo cp ./configs/systemd/mlapi.service /etc/systemd/system
sudo chmod 644 /etc/systemd/system/mlapi.service
sudo systemctl daemon-reload
# --now also starts the service while enabling it to run on boot
sudo systemctl enable mlapi.service --now
```