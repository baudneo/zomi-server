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
>[!TIP]
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

>[!TIP]
> `--dry-run` will show the commands that will be run without actually running them.