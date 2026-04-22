# Hugging Face kernel hub example

This shows an example using the Hugging Face kernel hub to download a
pre-compiled kernel.  In this example, the 
[activation](https://huggingface.co/kernels/kernels-community/activation)
gelu_and_mul kernel is demonstrated.  At the time of this writing, this set
of kernels has builds for Nvidia CUDA and Apple Metal.  For the exact set of
supported hardware, check the kernel card.

## Build

There is a Containerfile that encapsulates the environment needed to run a
torch application that uses the kernels interface on Nvidia GPUs.

```bash
podman build . -t gelu:latest
```

You can also see what the Containerfile does and recreate it locally in a
python virtual environment if you prefer not to use containers, for example
if you want to try it on Apple Metal.

## Run

Set the HF_TOKEN environment variable to your Hugging Face token.  Then:

```bash
podman run -it --rm --device nvidia.com/gpu=all --security-opt=label=disable -e HF_TOKEN=${HF_TOKEN} gelu:latest python3.12 gelu-and-mul-test.py
```

The output should look something like this if you have a supported Nvidia GPU
and >=580 driver:

```bash
==========
== CUDA ==
==========

CUDA Version 13.0.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Fetching 6 files: 100%|███████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  3.99it/s]
Download complete: : 4.18MB [00:01, 3.30MB/s]              Success!                   | 2/6 [00:01<00:03,  1.17it/s]
Download complete: : 4.18MB [00:01, 2.62MB/s]
```

Note that the kernels package will output progress information about the
download to stderr.  You can filter that out.  For example, you can
redirect the stderr to /dev/null if you want cleaner output.  Just be aware
that debugging failures will be harder that way.

```bash
$ podman run -it --rm --device nvidia.com/gpu=all --security-opt=label=disable -e HF_TOKEN=${HF_TOKEN} gelu:latest bash

==========
== CUDA ==
==========

CUDA Version 13.0.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[root@fc9f150b0e9e src]# python3.12 gelu-and-mul-test.py 2> /dev/null
Success!
```
