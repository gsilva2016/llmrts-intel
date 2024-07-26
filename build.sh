#!/bin/bash

echo "Building mlc-ai container"
docker build -t mlc-ai:1.0 .

echo "Building ipex-llm container"
docker build -t ipex-llm:1.0 -f Dockerfile.ipexllm . 

echo "Building ipex-xpu container"
docker build -t ipex-xpu:1.0 -f Dockerfile.ipex-xpu .
