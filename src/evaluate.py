import torch
import numpy as np
from tqdm import tqdm
from utils import non_max_suppression, convert_yolo_to_boxes, mean_average_precision

def calculate_iou(pred_mask, target_mask):
    """
    Calculate IoU for segmentation masks
    
    Args:
        pred_mask: tensor of shape (N, C, H, W)
        target_mask: tensor of shape (N, C, H, W)
    """
    intersection = torch.logical_and(pred_mask, target_mask).sum(dim=(2, 3))
    union = torch.logical_or(pred_mask, target_mask).sum(dim=(2, 3))
    
    # Add small epsilon to avoid division by zero
    iou = (intersection + 1e-8) / (union + 1e-8)
    
    return iou

def evaluate_model(model, data_loader, device, args):
    """
    Evaluate model on detection and segmentation tasks
    
    Args:
        model: YOLOUNet model
        data_loader: DataLoader for test set
        device: torch device
        args: Argument object containing evaluation parameters
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metric lists
    all_gt_boxes = []
    all_gt_classes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    
    # IoU metrics for segmentation
    iou_per_class = torch.zeros(args.num_classes + 1).to(device)  # +1 for background
    valid_pixels_per_class = torch.zeros(args.num_classes + 1).to(device)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get data
            images = batch['image'].to(device)
            detection_targets = batch['detection'].to(device)
            segmentation_targets = batch['segmentation'].to(device)
            
            batch_size = images.size(0)
            
            # Forward pass
            outputs = model(images)
            
            # Process detection predictions and targets
            for b in range(batch_size):
                # Process ground truth boxes
                gt_boxes, gt_classes = convert_yolo_to_boxes(
                    detection_targets[b],
                    grid_size=args.grid_size,
                    num_boxes=args.num_boxes,
                    num_classes=args.num_classes,
                    img_size=args.img_size
                )
                
                # Process predicted boxes
                pred_boxes, pred_scores, pred_classes = non_max_suppression(
                    outputs['detection'][b],
                    grid_size=args.grid_size,
                    num_boxes=args.num_boxes,
                    num_classes=args.num_classes,
                    img_size=args.img_size,
                    conf_threshold=args.conf_threshold,
                    nms_threshold=args.nms_threshold
                )
                
                # Append to lists
                if len(gt_boxes) > 0:
                    all_gt_boxes.append(gt_boxes)
                    all_gt_classes.append(gt_classes)
                else:
                    all_gt_boxes.append(torch.zeros((0, 4), device=device))
                    all_gt_classes.append(torch.zeros((0), dtype=torch.int64, device=device))
                
                if len(pred_boxes) > 0:
                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_pred_classes.append(pred_classes)
                else:
                    all_pred_boxes.append(torch.zeros((0, 4), device=device))
                    all_pred_scores.append(torch.zeros((0), device=device))
                    all_pred_classes.append(torch.zeros((0), dtype=torch.int64, device=device))
            
            # Process segmentation predictions
            pred_seg = outputs['segmentation']
            pred_seg = torch.sigmoid(pred_seg)
            pred_seg_binary = (pred_seg > 0.5)
            
            # Calculate IoU for each class
            for c in range(args.num_classes + 1):  # +1 for background
                # Create binary masks for this class
                pred_mask_c = pred_seg_binary[:, c]
                target_mask_c = segmentation_targets[:, c] > 0.5
                
                # Calculate IoU for this class
                intersection = torch.logical_and(pred_mask_c, target_mask_c).sum(dim=(1, 2))
                union = torch.logical_or(pred_mask_c, target_mask_c).sum(dim=(1, 2))
                
                # Count valid pixels (where union > 0)
                valid_pixels = (union > 0).float()
                valid_pixels_sum = valid_pixels.sum()
                valid_pixels_per_class[c] += valid_pixels_sum
                
                # Calculate IoU only for valid pixels
                if valid_pixels_sum > 0:
                    iou = (intersection + 1e-8) / (union + 1e-8)
                    iou_per_class[c] += (iou * valid_pixels).sum()
    
    # Calculate mAP for detection
    mAP, mAP_50, mAP_75 = mean_average_precision(
        all_pred_boxes,
        all_pred_scores,
        all_pred_classes,
        all_gt_boxes,
        all_gt_classes,
        num_classes=args.num_classes,
        device=device
    )
    
    # Calculate mean IoU for segmentation
    valid_classes = valid_pixels_per_class > 0
    mean_iou_per_class = torch.zeros_like(iou_per_class)
    mean_iou_per_class[valid_classes] = iou_per_class[valid_classes] / valid_pixels_per_class[valid_classes]
    mean_iou = mean_iou_per_class[valid_classes].mean().item()
    
    # Create metrics dictionary
    metrics = {
        'mAP': mAP,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
        'mean_iou': mean_iou
    }
    
    # Add per-class IoU
    for c in range(args.num_classes + 1):
        metrics[f'iou_class_{c}'] = mean_iou_per_class[c].item()
    
    return metrics
