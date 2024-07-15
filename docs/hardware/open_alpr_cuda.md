# Build OpenALPR with CUDA support
You can build OpenALPR with CUDA support following the example in [this script](https://github.com/ShinobiCCTV/Shinobi/blob/dev/INSTALL/openalpr-gpu-easy.sh)

# Config files to enable GPU
Don't forget to create 2 copies of the alpr config file. One for CPU and one for GPU.

If the git repo for openalpr is at `/opt/openalpr` this would be the commands to create the config files:

```bash
cp /opt/openalpr/config/openalpr.conf.defaults /etc/openalpr/openalpr.conf \
&& cp /tmp/etc/openalpr/openalpr.conf /etc/openalpr/openalpr.conf.gpu \
&& sed -i 's/detector =.*/detector = lbpgpu/g' /etc/openalpr/openalpr.conf.gpu
```