# Exercises Projects in learning [ECE408](https://wiki.illinois.edu/wiki/display/ECE408/Class+Schedule)
ECE408: Applied parallel programming is a course teaches many parallel programming concepts and its implements as well as programming, which is illustrated by CUDA. And there is also a TextBook called [Programming massively parallel processors](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) for this lesson. So this project is the mass practice and labs programsthat I wrote for this lesson and this book.

# How to run it?
## requirements
### Nvidia drivers
Nvidia drivers (non-free) (which is nvidia-dkms on my computer because I use linux-zen kernel, however if you run the linux or linux-lts kernel, this should be nvidia. anyway, follow the tutorial of archwiki.), cuda, cuda-tools.

After install these packages, we can test the environment by:

``` sh
$ nvidia-smi
```

Its output is as follow (truncated), note the CUDA Version:
``` text
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.41.03              Driver Version: 530.41.03    CUDA Version: 12.1     |
....
```

``` sh
$ nvcc -V
```

Its output:
``` text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

Make sure these two cuda versions are identical, (12.1 in my case).

### docker & nvidia-container
Because of a [bug from archlinux upstream](https://bugs.archlinux.org/task/78362), we have to use docker as a workaround to compile cuda code. Although this is an workaround as original purpose, it quite provide a stable environment for us.

We can setup our nvidia docker environment by 3 different ways, user can check it on archwiki:docker by searching nvidia.

``` sh
$ sudo pacman -S docker docker-compose nvidia-container-toolkit # or nvidia-container-runtime, its respect to your choice. 
```

If at the former step, your CUDA version is not 12.1, you must change the tag of `nvidia/cuda` in `Dockerfile`, says, 11.1, you can modify `Dockerfile` to:

``` dockerfile
FROM nvidia/cuda:11.1.1-devel-ubi8 # It just an example, i don't know if nvidia/cuda have this tag or not, please check it on dockerhub.
RUN dnf install -y cmake 
```

## Launch
### If you are Chinese
我用的是 Clash 进行代理，所以默认的端口号是 7890，如果你和我一样使用 Clash，就可以跳过这步。否则将端口号改成你的 Proxy 端口。
### If you are not Chinese
Remove all `proxy` stuff in `Dockerfile` and `docker-compose.yaml` like this:

`Dockerfile`:
``` dockerfile
FROM nvidia/cuda:12.1.1-devel-ubi8
RUN dnf install -y cmake
```

`docker-compose.yaml`:
``` yaml
# yaml 配置
services:
  cuda:
    build: .
    command: [ "bash", "./run.sh" ]
    volumes:
      - .:${DIR}:rw
    working_dir: ${DIR}
version: '3'
```

### Finally
Once requirements be set up, reader can build the container used by this project by:

``` sh
$ make docker-image
```

And build the project and test it by:

``` sh
$ make
```
