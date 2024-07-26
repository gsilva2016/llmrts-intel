# Evaluate Inference Runtimes for LLM/DL Models and Intel Hardware

# Build Containers

Build all containers for MLC-AI, IPEX-LLM, and IPEX-XPU.

./build.sh

## MLC-AI LLM

Run Qwen2 0.5B MLC-AI LLM inference using XPU via Vulkan. Metrics available via /stats. Use vulkaninfo to find the correct GPU ID to specify (current default GPU device is vulkan:0).


```
./run-mlcai.sh
```

Run Qwen2 0.5B q0f16

```
./qwen2-0.5B-Instruct-mlcllm-intel.sh
```

Run Qwen2 0.5B q4f16

```
./qwen2-0.5B-Instruct-mlcllm-q4fp16-intel.sh
```

## IPEX-LLM

Run Qwen2 0.5B torch infernece using XPU via OpenCL compute. Use clinfo to find the correct GPU ID to specify (current default GPU device is xpu:0).

```
./run-ipexllm.sh
```

Run Qwen2 0.5B q0f16

```
python3 qwen2-0.5B-Instruct-ipexllm.py
```

Run Qwen2 0.5B q4f16

```
python3 qwen2-0.5B-Instruct-q4fp16-ipexllm.py
```

## IPEX-XPU

Run non-LLM based torch inference with Yolov5 and XPU via OpenCL compute. Use clinfo to find the correct GPU ID to specify (current default GPU device is xpu:0).

```
./run-ipex-xpu.sh
```

Run Yolov5n FP32

```
python3 yolov5-torch-xpu.py
```
