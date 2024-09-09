import cv2
import torch

# Color dictionary
COLOR_DICT = {
    'person': (0, 255,0), 'bicycle': (72, 61, 139), 'car': (47, 79, 79), 'motorcycle': (0, 100, 0),
    'airplane': (139, 0, 139), 'bus': (85, 107, 47), 'train': (184, 134, 11), 'truck': (139, 69, 19),
    'boat': (0, 0, 128), 'traffic light': (255, 69, 0), 'fire hydrant': (255, 140, 0),
    'stop sign': (255, 215, 0), 'parking meter': (240, 230, 140), 'bench': (139, 0, 0),
    # Add the rest of your COLOR_DICT entries here...
    '__default__': (255, 0, 255)  # Default color
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    # Add the rest of your categories here...
]

def get_predictions(img, model, transform, threshold=0.7):
    img_resized = cv2.resize(img, (640, 480))
    img_tensor = transform(img_resized).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)

    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].numpy()]
    pred_scores = outputs[0]['scores'].detach().numpy()
    pred_boxes = outputs[0]['boxes'].detach().numpy()

    pred_boxes = pred_boxes[pred_scores >= threshold].astype(int)
    pred_classes = [pred_classes[i] for i in range(len(pred_scores)) if pred_scores[i] >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]

    pred_boxes = (pred_boxes * [img.shape[1] / 640, img.shape[0] / 480, img.shape[1] / 640, img.shape[0] / 480]).astype(int)

    return pred_boxes, pred_classes, pred_scores
