from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import torch
import cv2
import numpy as np
from PIL import Image
import time


# Assuming `utils` has been adapted for PyTorch or provides framework-agnostic functions
import core.utils_v2 as utils

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

model = YOLO('./yolov8_weight/best.pt')

# model.eval()  # Set the model to evaluation mode

obj_classes = ["Apple", "Facebook", "Samsung", "Microsoft", "Google", "Others"]
num_classes = 6

# YOLO DETECTION for YOLOv8 --------------------

def detection(vid):
    global model, obj_classes
    
    with torch.no_grad():  # Disable gradient calculation for inference
    
        return_value, frame = vid.read()
        if not return_value:
            raise ValueError("No image!")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Preprocess the image for the model; ensure this matches your model's requirements
        image_data = preprocess_image(frame_rgb, [640, 640])

        # Run the model on the preprocessed image
        pred_bbox = model(image_data, conf=0.65)
        print(type(pred_bbox))
        print(pred_bbox)

        print('here')

        # Format the model's output for further processing
        formatted_boxes = format_boxes(pred_bbox, obj_classes)

        # Draw bounding boxes on the original image
        result_image = draw_bboxes(frame_rgb, formatted_boxes, obj_classes)
        
        # Convert the result back to BGR for OpenCV compatibility
        result = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        
        # Count detected objects by class
        # Convert formatted_boxes to a numpy array
        formatted_boxes_array = np.array(formatted_boxes)

        # Check if formatted_boxes_array is not empty and is two-dimensional
        if formatted_boxes_array.ndim == 2 and formatted_boxes_array.size > 0:
            class_count = [np.sum(formatted_boxes_array[:, -1] == i) for i in range(len(obj_classes))]
        else:
            # Handle the case where formatted_boxes is empty or not as expected
            class_count = [0 for _ in range(len(obj_classes))]

        
        return result, class_count

def format_boxes(pred_bbox, obj_classes):
    formatted_boxes = []

    if pred_bbox:
        # Assuming the first element in pred_bbox list for a single image scenario
        results = pred_bbox[0]
        
        # Directly access the 'xyxy', 'conf', and 'cls' attributes of the 'boxes'
        if hasattr(results.boxes, 'xyxy') and hasattr(results.boxes, 'conf') and hasattr(results.boxes, 'cls'):
            # Convert tensors to numpy arrays if they are not already
            xyxy = results.boxes.xyxy.cpu().numpy() if isinstance(results.boxes.xyxy, torch.Tensor) else results.boxes.xyxy
            conf = results.boxes.conf.cpu().numpy() if isinstance(results.boxes.conf, torch.Tensor) else results.boxes.conf
            cls = results.boxes.cls.cpu().numpy() if isinstance(results.boxes.cls, torch.Tensor) else results.boxes.cls
            
            # Iterate through each box and format the bounding box information
            for i in range(xyxy.shape[0]):  # Assuming xyxy is a 2D array with shape [num_boxes, 4]
                x1, y1, x2, y2 = xyxy[i]
                score = conf[i]
                class_id = cls[i]
                formatted_boxes.append([x1, y1, x2, y2, score, class_id])

    return formatted_boxes





def preprocess_image(image, target_size):
    # Adapted preprocessing to match your model's requirements
    # Include resizing and normalization steps as necessary
    return utils.image_preprocess(image, target_size)

def draw_bboxes(image, bboxes, obj_classes):
    # Utilize your existing drawing function or adapt it as necessary
    return utils.draw_bounding_boxes(image, bboxes, obj_classes=obj_classes, show_label=True)


