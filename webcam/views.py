from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import torch
import cv2
import numpy as np
from PIL import Image
import time

# Assuming `utils` has been adapted for PyTorch or provides framework-agnostic functions
import core.utils_copy as utils

from ultralytics import YOLO


# HOME PAGE -------------------------
def index(request):
	template = loader.get_template('index.html')
	return HttpResponse(template.render({}, request))
# -----------------------------------

# CAMERA 1 PAGE ---------------------
def camera_1(request):
	template = loader.get_template('camera1.html')
	return HttpResponse(template.render({}, request))
# -----------------------------------

# DISPLAY CAMERA 1 ------------------
def stream_1():

	cam_id = 0
	vid = cv2.VideoCapture(cam_id)

	while True:
		frame, class_count = detection(vid)

		frame = cv2.resize(frame, (1000, 700))

		print("\nObjects in frame:")
		row = 0
		for k in range(len(class_count)):
			if class_count[k] > 0: 
				row += 1
				infor = str(obj_classes[k]) + ": " + str(int(class_count[k]))
				print("  " + infor)
				frame = cv2.putText(frame,infor,(20,(row+1)*35), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

		cv2.imwrite('currentframe.jpg', frame)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + open('currentframe.jpg', 'rb').read() + b'\r\n')

def video_feed_1(request):
	return StreamingHttpResponse(stream_1(), content_type='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------

# Load the YOLOv8 model
# model_path = '../yolov8_weight/best.pt'
# model = torch.jit.load('./yolov8_weight/best.pt')

model = YOLO('./yolov8_weight/best.pt')

# model.eval()  # Set the model to evaluation mode

obj_classes = ["Apple", "Facebook", "Samsung", "Microsoft", "Google", "Others"]
num_classes = 6

# YOLO DETECTION for YOLOv8 --------------------
def detection(vid):
    global model, obj_classes  # Assuming 'model' and 'obj_classes' are defined globally or accessible otherwise
    
    with torch.no_grad():  # Disable gradient calculation for inference
        return_value, frame = vid.read()
        if not return_value:
            raise ValueError("No image!")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preprocess(np.copy(frame_rgb), [640, 640])

        pred_bbox = model(image_data)

        formatted_boxes = []
        print("Sample of formatted boxes:", formatted_boxes[:5])


        if hasattr(pred_bbox, 'boxes') and len(pred_bbox.boxes):
            for box in pred_bbox.boxes:
                x1, y1, x2, y2, score, class_id = box.xyxy[0], box.xyxy[1], box.xyxy[2], box.xyxy[3], box.confidence, box.class_id
                formatted_boxes.append([x1, y1, x2, y2, score, class_id])

        formatted_boxes = np.array(formatted_boxes) if formatted_boxes else np.empty((0, 6))

        bboxes = utils.postprocess_boxes(formatted_boxes, frame.shape[:2], 640, 0.3) if formatted_boxes.size else formatted_boxes
        print("Boxes after post-processing:", bboxes)

        bboxes = utils.nms(bboxes, 0.45, method='nms') if bboxes.size else bboxes

        image, detected = utils.draw_bbox(frame_rgb.copy(), bboxes) if bboxes.size else (frame_rgb.copy(), np.empty((0, 6)))

        if detected.size:
            class_count = [np.sum(detected[:, 5] == i) for i in range(len(obj_classes))]
        else:
            class_count = [0 for _ in range(len(obj_classes))]

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return result, class_count


# def detection(vid):
#     global model, obj_classes  # Assuming 'model' and 'obj_classes' are defined globally or accessible otherwise
    
#     with torch.no_grad():  # Disable gradient calculation for inference
#         return_value, frame = vid.read()
#         if not return_value:
#             raise ValueError("No image!")

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image_data = utils.image_preprocess(np.copy(frame_rgb), [640, 640])

#         pred_bbox = model(image_data)

#         formatted_boxes = []

#         if hasattr(pred_bbox, 'boxes') and len(pred_bbox.boxes):
#             for box in pred_bbox.boxes:
#                 x1, y1, x2, y2, score, class_id = box.xyxy[0], box.xyxy[1], box.xyxy[2], box.xyxy[3], box.confidence, box.class_id
#                 formatted_boxes.append([x1, y1, x2, y2, score, class_id])

#         formatted_boxes = np.array(formatted_boxes) if formatted_boxes else np.empty((0, 6))

#         bboxes = utils.postprocess_boxes(formatted_boxes, frame.shape[:2], 640, 0.3) if formatted_boxes.size else formatted_boxes
#         bboxes = utils.nms(bboxes, 0.45, method='nms') if bboxes.size else bboxes

#         image, detected = utils.draw_bbox(frame_rgb.copy(), bboxes) if bboxes.size else (frame_rgb.copy(), np.empty((0, 6)))

#         if detected.size:
#             class_count = [np.sum(detected[:, 5] == i) for i in range(len(obj_classes))]
#         else:
#             class_count = [0 for _ in range(len(obj_classes))]

#         result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         return result, class_count