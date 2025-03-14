import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utils import non_max_suppression, convert_yolo_to_boxes

def generate_colors(num_classes):
    """Generate different colors for each class"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors

def visualize_predictions(image, det_output, seg_output, class_names, grid_size, num_boxes, num_classes, 
                          conf_threshold=0.5, nms_threshold=0.4, save_path=None):
    """
    Visualize detection and segmentation predictions
    
    Args:
        image: tensor of shape (3, H, W)
        det_output: tensor of shape (grid_size, grid_size, 5*num_boxes + num_classes)
        seg_output: tensor of shape (num_classes+1, H, W)
        class_names: list of class names
        grid_size: Grid size for YOLO detection
        num_boxes: Number of bounding boxes per grid cell
        num_classes: Number of object classes
        conf_threshold: Confidence threshold for detection
        nms_threshold: NMS threshold for detection
        save_path: Path to save visualization
    """
    # Generate colors
    colors = generate_colors(num_classes)
    
    # Convert image to numpy (0-255 RGB)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    # Get image dimensions
    img_h, img_w = img_np.shape[:2]
    
    # Apply NMS to get detection results
    boxes, scores, class_ids = non_max_suppression(
        det_output,
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
        img_size=img_h,  # Assuming square image
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    # Create segmentation mask
    seg_output = torch.sigmoid(seg_output)
    seg_mask = torch.argmax(seg_output, dim=0).cpu().numpy()
    
    # Create segmentation visualization
    seg_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Add colors to segmentation mask (skip background class 0)
    for c in range(1, num_classes + 1):
        seg_vis[seg_mask == c] = colors[c - 1]
    
    # Blend segmentation with original image
    alpha = 0.5
    blended = cv2.addWeighted(img_np, alpha, seg_vis, 1 - alpha, 0)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Show blended image
    ax.imshow(blended)
    
    # Add detection boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
        class_id = int(class_ids[i].item())
        score = scores[i].item()
        
        #