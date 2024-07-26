import torch
from openvino.runtime import Core
import openvino.torch
import cv2

ie = Core()
# Use if device GPU or AUTO && GPU_AVAIL
ovconfig = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES", "PERFORMANCE_HINT": "LATENCY"}
model_name = "yolov5n"

model = torch.hub.load("ultralytics/yolov5", model_name)

# Reference: https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html
model = torch.compile(model, backend='openvino', options = { "device": "GPU", "config": ovconfig })
#output_layer = model.output(0)

source = "/dev/video2" 
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames_avail, frame = cap.read()
while frames_avail:
    with torch.no_grad():
        results = model(frame)
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    frames_avail, frame = cap.read()

