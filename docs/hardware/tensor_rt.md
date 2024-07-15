# Build Tensor RT
>[!IMPORTANT]
> When you compile models for Tensor RT, you must compile them for the specific hardware you are using.
> Meaning, if you compile a model for a Jetson Nano, it will not work on a Jetson Xavier or a desktop GPU.
> The same goes for the other way around.

> [!NOTE]
> Whatever version of Tensor RT you use to compile the model, you must use the same version of 
> Tensor RT to run the model to ensure compatibility.

There are guides available online on how to compile models for Tensor RT. 
[Here is the official Nvidia TAR file docs](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)