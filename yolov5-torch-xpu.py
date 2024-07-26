import torch
import intel_extension_for_pytorch as ipex
import cv2

model_name = "yolov5n"

model = torch.hub.load("ultralytics/yolov5", model_name)
model.eval()
model = model.to("xpu")
model = ipex.optimize(model)


source = "/dev/video2" 
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames_avail, frame = cap.read()
while frames_avail:
    with torch.no_grad():
        results = model(frame)
        torch.xpu.synchronize()
        results.print()

    frames_avail, frame = cap.read()
