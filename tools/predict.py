import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps
import os
import csv
import numpy as np
from torchvision.transforms import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(weights_file):
    """
    Load model with given weights
    """
    if not weights_file:
        raise FileNotFoundError(f"No weiths to load")

    # Initialize the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, min_size=600, max_size=1000)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes=2)
    print(f"Loading weights: {weights_file}")
    
    # Load the checkpoint
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    return model

def predict(image_path, output_path, weights_file, confidence_threshold=0.5):
    """
    Predict boxes for a given box and save the results in a given CSV file
    Args:
        image_path: image path OR list of image paths to predict
        output_path: path to output CSV file
        weights_file: path to model weights file
        confidence_threshold: confidence score threshold to validate output box from model
    """
    # Check if CSV file exists
    if not os.path.exists(output_path):
        with open(output_path, mode='w') as f:
            f.write('image_name,scroll_number,xmin,ymin,xmax,ymax\n')
        print(f"Created file: {output_path}")
    else:
        print(f"The file {output_path} already exists")

    # Load the model
    model = load_model(weights_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load and prepare the image(s)
    if isinstance(image_path, str):
        image_paths = [image_path]
    elif isinstance(image_path, list):
        image_paths = image_path
    else:
        raise ValueError("Must pass path to image or list of paths")
    
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.basename(image_path)
        image = ImageOps.exif_transpose(image)
        # Convert to tensor
        image_tensor = F.to_tensor(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Make the inference
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Extract the predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
    
        # Filter by confidence score
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
    
        # Calculate the scroll_numbers (distance to the top left corner)
        scroll_numbers = np.sqrt(boxes[:, 0]**2 + boxes[:, 1]**2)
        
        # Sort by scroll_number
        sorted_indices = np.argsort(scroll_numbers)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]
        scroll_numbers = scroll_numbers[sorted_indices]
        
        # Save in the CSV file
        with open(output_path, 'a') as f:
            writer = csv.writer(f)
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                writer.writerow([
                    image_name,
                    f"{i+1}",
                    f"{xmin:.0f}",
                    f"{ymin:.0f}",
                    f"{xmax:.0f}",
                    f"{ymax:.0f}"
                ])
    
    print(f"Predictions saved in {output_path}")