FROM nvcr.io/nvidia/cuda:13.0.3-cudnn-devel-ubi9


RUN dnf update -y && \
    dnf install -y python3.12 python3.12-pip python3.12-devel vim

RUN pip3.12 install --upgrade pip
RUN pip3.12 install uv

WORKDIR /src

# Create virtual env
RUN uv venv venv --python 3.12
ENV PATH="/src/venv/bin:/src/venv/lib64/python3.12/site-packages/nvidia/cu13/bin:$PATH"
ENV VIRTUAL_ENV=/src/venv
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.11 torchvision kernels

COPY gelu-and-mul-test.py /src/

