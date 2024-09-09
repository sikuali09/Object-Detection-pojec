import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_model():
    # Load pre-trained Faster-RCNN model
    model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
    model.eval()  # Set the model to evaluation mode
    return model

def get_transform():
    return T.Compose([T.ToTensor()])  # Define the transformation
