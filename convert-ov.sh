#!/bin/bash

python3 convert-ov.py --model_id qwen/Qwen2-0.5B-Instruct --precision int4 --output ./Qwen2-0.5B-q4f16-Instruct-ov --modelscope
python3 convert-ov.py --model_id qwen/Qwen2-0.5B-Instruct --precision fp16 --output ./Qwen2-0.5B-q0f16-Instruct-ov --modelscope
