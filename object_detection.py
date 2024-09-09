import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn


model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT") # Load pre-trained model
model.eval()


transform = T.Compose([T.ToTensor()]) # torch transfromation 

# color dictionary
COLOR_DICT = {
    'person': (0, 255,0),       # Dark Red
    'bicycle': (72, 61, 139),    # Dark Slate Blue
    'car': (47, 79, 79),         # Dark Slate Gray
    'motorcycle': (0, 100, 0),   # Dark Green
    'airplane': (139, 0, 139),   # Dark Magenta
    'bus': (85, 107, 47),        # Dark Olive Green
    'train': (184, 134, 11),     # Dark Goldenrod
    'truck': (139, 69, 19),      # Dark Saddle Brown
    'boat': (0, 0, 128),         # Navy
    'traffic light': (255, 69, 0),# Orange Red
    'fire hydrant': (255, 140, 0),# Dark Orange
    'stop sign': (255, 215, 0),  # Gold
    'parking meter': (240, 230, 140),# Khaki
    'bench': (139, 0, 0),        # Dark Red
    'bird': (199, 21, 133),      # Medium Violet Red
    'cat': (219, 112, 147),      # Pale Violet Red
    'dog': (255, 20, 147),       # Deep Pink
    'horse': (218, 112, 214),    # Orchid
    'sheep': (186, 85, 211),     # Medium Orchid
    'cow': (153, 50, 204),       # Dark Orchid
    'elephant': (148, 0, 211),   # Dark Violet
    'bear': (138, 43, 226),      # Blue Violet
    'zebra': (160, 32, 240),     # Purple
    'giraffe': (147, 112, 219),  # Medium Purple
    'backpack': (123, 104, 238), # Medium Slate Blue
    'umbrella': (106, 90, 205),  # Slate Blue
    'handbag': (72, 61, 139),    # Dark Slate Blue
    'tie': (75, 0, 130),         # Indigo
    'suitcase': (238, 130, 238), # Violet
    'frisbee': (221, 160, 221),  # Plum
    'skis': (218, 112, 214),     # Orchid
    'snowboard': (186, 85, 211), # Medium Orchid
    'sports ball': (153, 50, 204),# Dark Orchid
    'kite': (148, 0, 211),       # Dark Violet
    'baseball bat': (138, 43, 226),# Blue Violet
    'baseball glove': (160, 32, 240),# Purple
    'skateboard': (147, 112, 219),# Medium Purple
    'surfboard': (123, 104, 238),# Medium Slate Blue
    'tennis racket': (106, 90, 205),# Slate Blue
    'bottle': (70, 61, 139),     # Dark Slate Blue
    'wine glass': (75, 0, 130),  # Indigo
    'cup': (238, 130, 238),      # Violet
    'fork': (221, 160, 221),     # Plum
    'knife': (218, 112, 214),    # Orchid
    'spoon': (186, 85, 211),     # Medium Orchid
    'bowl': (153, 50, 204),      # Dark Orchid
    'banana': (148, 0, 211),     # Dark Violet
    'apple': (138, 43, 226),     # Blue Violet
    'sandwich': (160, 32, 240),  # Purple
    'orange': (147, 112, 219),   # Medium Purple
    'broccoli': (123, 104, 238), # Medium Slate Blue
    'carrot': (106, 90, 205),    # Slate Blue
    'hot dog': (72, 61, 139),    # Dark Slate Blue
    'pizza': (75, 0, 130),       # Indigo
    'donut': (238, 130, 238),    # Violet
    'cake': (221, 160, 221),     # Plum
    'chair': (0, 0, 255),    # Orchid
    'couch': (186, 85, 211),     # Medium Orchid
    'potted plant': (153, 50, 204),# Dark Orchid
    'bed': (148, 0, 211),        # Dark Violet
    'dining table': (138, 43, 226),# Blue Violet
    'toilet': (160, 32, 240),    # Purple
    'tv': (147, 112, 219),       # Medium Purple
    'laptop': (0, 0, 255),   # Medium Slate Blue
    'mouse': (106, 90, 205),     # Slate Blue
    'remote': (72, 61, 139),     # Dark Slate Blue
    'keyboard': (75, 0, 130),    # Indigo
    'cell phone': (238, 130, 238),# Violet
    'microwave': (221, 160, 221),# Plum
    'oven': (218, 112, 214),     # Orchid
    'toaster': (186, 85, 211),   # Medium Orchid
    'sink': (153, 50, 204),      # Dark Orchid
    'refrigerator': (148, 0, 211),# Dark Violet
    'book': (138, 43, 226),      # Blue Violet
    'clock': (160, 32, 240),     # Purple
    'vase': (147, 112, 219),     # Medium Purple
    'scissors': (123, 104, 238), # Medium Slate Blue
    'teddy bear': (106, 90, 205),# Slate Blue
    'hair drier': (72, 61, 139), # Dark Slate Blue
    'toothbrush': (75, 0, 130),  # Indigo
    # Default color for others
    '__default__': (255, 0, 255) # Magenta
}

def get_predictions(img, threshold=0.7):
    # Resize image
    img_resized = cv2.resize(img, (640, 480))
    img_tensor = transform(img_resized)
    img_tensor = img_tensor.unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Filter predictions (based on  threshold)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].numpy()]
    pred_scores = outputs[0]['scores'].detach().numpy()
    pred_boxes = outputs[0]['boxes'].detach().numpy()
    pred_boxes = pred_boxes[pred_scores >= threshold].astype(int)
    pred_classes = [pred_classes[i] for i in range(len(pred_scores)) if pred_scores[i] >= threshold]
    pred_scores = pred_scores[pred_scores >= threshold]

   
    pred_boxes = (pred_boxes * [img.shape[1] / 640, img.shape[0] / 480, img.shape[1] / 640, img.shape[0] / 480]).astype(int)  # Adjust boxes to original image size

    return pred_boxes, pred_classes, pred_scores

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def live_object_detection():
    cap = cv2.VideoCapture(0)
    
    # Set camera properties (e.g., frame width, height, and FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    cap.set(cv2.CAP_PROP_FPS, 8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes, scores = get_predictions(frame, threshold=0.7)

        for box, label, score in zip(boxes, classes, scores):
            color = COLOR_DICT.get(label, COLOR_DICT['__default__'])  # Use specific color/default to magenta
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Live Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_object_detection()   # infinite loop
