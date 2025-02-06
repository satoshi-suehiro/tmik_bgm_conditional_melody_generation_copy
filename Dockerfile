FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

WORKDIR /workspace/app
COPY . /workspace/app/

RUN apt update && apt install -y fluidsynth

RUN pip install --upgrade pip \
    && pip install uv==0.5.26 \
    && uv sync

