import cv2
import torch
import numpy as np

# Assuming `model` is your loaded YOLOv8 model and `obj_classes` are the class names
# Also assuming necessary utilities for image processing are correctly set up

def image_preprocess(image, target_size=(640, 640)):
    """
    Preprocess the image for YOLOv8 detection.
    """
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full((ih, iw, 3), 128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.0  # Normalize

    # Convert to PyTorch tensor
    image_paded = np.transpose(image_paded, (2, 0, 1))
    image_paded = torch.from_numpy(image_paded).float().unsqueeze(0)  # Add batch dimension
    return image_paded

def draw_bounding_boxes(image, bboxes, obj_classes, show_label=True):
    for bbox in bboxes:
        if len(bbox) < 6:
            continue
        x1, y1, x2, y2, score, class_id = bbox
        class_id = int(class_id)
        if class_id >= len(obj_classes) or class_id < 0:
            continue

        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        if show_label:
            label = f"{obj_classes[class_id]}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (int(x1), int(y1) - text_height - baseline), 
                          (int(x1) + text_width, int(y1)), (0, 255, 0), thickness=cv2.FILLED)
            cv2.putText(image, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 2)

    return image





def detect_and_draw(image):
    """
    Perform object detection and draw bounding boxes around detected objects.
    """
    global model, obj_classes

    # Preprocess
    input_image = image_preprocess(image)

    # Detection
    with torch.no_grad():
        detections = model(input_image)

    # Assume detections are formatted properly after postprocessing (x1, y1, x2, y2, score, class_id)
    # Postprocess (simplified, assuming you have a function that does it)
    detections = postprocess_boxes(detections.cpu().numpy(), image.shape[:2], 640, 0.3)

    # Draw bounding boxes
    result_image = draw_bounding_boxes(image.copy(), detections, obj_classes)

    return result_image

# You might need to adjust `postprocess_boxes` according to the actual output format of your YOLOv8 model.



def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    pred_bbox = np.array(pred_bbox) if isinstance(pred_bbox, list) else pred_bbox
    if pred_bbox.size == 0:
        print("No predictions to process.")
        return np.array([])

    # Check if predictions are empty
    if len(pred_bbox) == 0 or pred_bbox.ndim < 2 or pred_bbox.shape[1] < 5:
        # Return an empty array if no detections or pred_bbox not as expected
        return np.array([])
    
    valid_scale = [0, np.inf]
    
    # Assuming pred_bbox is already a np.array or has been converted earlier
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # Convert (x, y, w, h) to (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # Adjust coordinates to match the original image size
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # Clip boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    
    # Discard invalid boxes
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # Discard some invalid boxes based on scale
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


    
    # Debug scale masking
    print("Bboxes scale before filtering:", bboxes_scale)

    # Apply score threshold
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    print("Scores before thresholding:", scores)

    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)

    # Select boxes that meet criteria
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)




