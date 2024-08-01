# Evaluate Inference Runtimes for LLM/DL Models on Intel Hardware

# Build Containers

Build all containers for MLC-AI, IPEX-LLM, and IPEX-XPU.

```
./build.sh
```

## MLC-AI LLM Inference

Run Qwen2 0.5B MLC-AI LLM inference using XPU via Vulkan. Metrics available via /stats. Use vulkaninfo to find the correct GPU ID to specify (current default GPU device is vulkan:0).


```
./run-mlcai.sh
```

Run Qwen2 0.5B q0f16

```
python3 qwen2-0.5B-Instruct-mlcllm-intel.py
```

Run Qwen2 0.5B q4f16

```
python3 qwen2-0.5B-Instruct-mlcllm-q4fp16-intel.py
```

## IPEX-LLM Inference

Run Qwen2 0.5B Intel® Extension for PyTorch* infernece using XPU via OpenCL compute. Use clinfo to find the correct GPU ID to specify (current default GPU device is xpu:0).

```
./run-ipexllm.sh
```

Run Qwen2 0.5B q0f16

```
python3 qwen2-0.5B-Instruct-ipexllm.py  -d <XPU:DEVICE_ID> e.g. xpu:0 or xpu:1
```

Run Qwen2 0.5B q4f16

```
python3 qwen2-0.5B-Instruct-q4fp16-ipexllm.py -d <XPU:DEVICE_ID> e.g. xpu:0 or xpu:1
```

## OpenVINO LLM Inference

Run Qwen2 0.5B Intel OpenVINO inference using GPU via OpenCL compute. Use clinfo to find the correct GPU ID to specify (current default GPU device is GPU.0).

```
./run-openvino.sh
```

Run Qwen2 0.5B q0f16

```
./qwen2-0.5B-Instruct-openvino-q4fp16.sh <GPU.ID> e.g. GPU.0 or GPU.1
```

Run Qwen2 0.5B q4f16

```
./qwen2-0.5B-Instruct-openvino-q4fp16.sh <GPU.ID> e.g. GPU.0 or GPU.1

```


## IPEX Yolov5 XPU Inference

Run non-LLM based Intel® Extension for PyTorch* inference with Yolov5 and XPU via OpenCL compute. Use clinfo to find the correct GPU ID to specify (current default GPU device is xpu:0).

```
./run-ipex-xpu.sh
```

Run Yolov5n FP32

```
python3 yolov5-torch-xpu.py
```
