FROM ubuntu:22.04


WORKDIR /app
USER root
ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_DEPENDENCIES="curl wget build-essential git clinfo vulkan-tools cmake python3-pip vim lsb-release software-properties-common gnupg git-lfs"
RUN apt -y update && \
    apt install -y ${BUILD_DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
RUN apt -y update && apt install -y wget && \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

RUN python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly

COPY qwen2-0.5B-Instruct-mlcllm-intel.sh .
COPY qwen2-0.5B-Instruct-mlcllm-q4fp16-intel.sh* .
