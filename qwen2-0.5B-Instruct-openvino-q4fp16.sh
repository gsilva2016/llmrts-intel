#!/bin/bash
python chat.py -m ./Qwen2-0.5B-q4f16-Instruct-ov/ --max_sequence_length 4096 --device $1
