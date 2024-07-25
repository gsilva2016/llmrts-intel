#!/bin/bash

echo "Building mlc-ai container"
docker build -t mlc-ai:1.0 .

echo "Building ipex-llm container"
docker build -t ipex-llm:1.0 -f Dockerfile.ipexllm .


#echo "Building itrex container"
#docker build -t itrex:1.0 -f Dockerfile.itrex .

