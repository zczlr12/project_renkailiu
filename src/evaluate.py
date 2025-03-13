from pathlib import Path
import json
import torch
from utils import YOLOUNet, MultiTaskLoss, non_max_suppression, \
    calculate_map, calculate_iou
import matplotlib.pyplot as plt
from dataloader import get_data_loaders


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class_names = ["Car", "Pedestrian", "Bicyclist", "MotorcycleScooter",
                   "Truck_Bus"]

    # Hyperparameters
    batch_size = 8
    grid_size = 7
    num_boxes = 19
    num_classes = 5

    # Dataset paths
    root_dir = Path(__file__).parents[1]

    _, _, test_loader, _ = get_data_loaders(
        root_dir=root_dir / "data",
        batch_size=batch_size,
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes,
    )

    # Initialize model
    model = YOLOUNet(
        grid_size=grid_size,
        num_boxes=num_boxes,
        num_classes=num_classes
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    criterion = MultiTaskLoss(lambda_coord=5.0, lambda_noobj=0.5, lambda_seg=1.0)
    
    all_pred_boxes = []
    all_target_boxes = []
    all_pred_masks = []
    all_target_masks = []
    
    with torch.no_grad():
        for batch_idx, (images, yolo_targets, seg_targets) in enumerate(test_loader):
            # Move data to device
            images = images.to(device)
            yolo_targets = yolo_targets.to(device)
            seg_targets = seg_targets.to(device)
            
            # Forward pass
            yolo_output, seg_output = model(images)
            
            # Calculate loss
            loss, _, _ = criterion(
                (yolo_output, seg_output),
                (yolo_targets, seg_targets)
            )
            
            test_loss += loss.item()
            
            # Process predictions for evaluation
            # Similar to validation processing
            batch_pred_boxes = []
            batch_target_boxes = []
            
            for i in range(images.size(0)):
                pred_boxes = []
                target_boxes = []
                
                # Process each grid cell
                for row in range(model.grid_size):
                    for col in range(model.grid_size):
                        # Box 1
                        pred_box1 = yolo_output[i, row, col, :5].clone()
                        confidence1 = pred_box1[4]
                        
                        if confidence1 > 0.1:
                            pred_box1[0] = (pred_box1[0] + col) / model.grid_size
                            pred_box1[1] = (pred_box1[1] + row) / model.grid_size
                            class_scores = yolo_output[i, row, col, 5*model.num_boxes:].clone()
                            class_idx = torch.argmax(class_scores).item()
                            pred_boxes.append(torch.cat([pred_box1, torch.tensor([class_idx], device=device)]))
                        
                        # Box 2 if applicable
                        if model.num_boxes > 1:
                            pred_box2 = yolo_output[i, row, col, 5:10].clone()
                            confidence2 = pred_box2[4]
                            
                            if confidence2 > 0.1:
                                pred_box2[0] = (pred_box2[0] + col) / model.grid_size
                                pred_box2[1] = (pred_box2[1] + row) / model.grid_size
                                class_scores = yolo_output[i, row, col, 5*model.num_boxes:].clone()
                                class_idx = torch.argmax(class_scores).item()
                                pred_boxes.append(torch.cat([pred_box2, torch.tensor([class_idx], device=device)]))
                        
                        # Target boxes
                        target_box1 = yolo_targets[i, row, col, :5].clone()
                        if target_box1[4] > 0:
                            target_box1[0] = (target_box1[0] + col) / model.grid_size
                            target_box1[1] = (target_box1[1] + row) / model.grid_size
                            target_class_scores = yolo_targets[i, row, col, 5*model.num_boxes:].clone()
                            target_class_idx = torch.argmax(target_class_scores).item()
                            target_boxes.append(torch.cat([target_box1[:4], torch.tensor([target_class_idx], device=device)]))
                        
                        if model.num_boxes > 1:
                            target_box2 = yolo_targets[i, row, col, 5:10].clone()
                            if target_box2[4] > 0:
                                target_box2[0] = (target_box2[0] + col) / model.grid_size
                                target_box2[1] = (target_box2[1] + row) / model.grid_size
                                target_class_scores = yolo_targets[i, row, col, 5*model.num_boxes:].clone()
                                target_class_idx = torch.argmax(target_class_scores).item()
                                target_boxes.append(torch.cat([target_box2[:4], torch.tensor([target_class_idx], device=device)]))
                
                # Apply NMS
                if pred_boxes:
                    pred_boxes = torch.stack(pred_boxes)
                    pred_boxes = non_max_suppression(pred_boxes)
                
                # Add to batch results
                batch_pred_boxes.extend(pred_boxes)
                if target_boxes:
                    batch_target_boxes.extend(target_boxes)
            
            # Add batch results to overall results
            all_pred_boxes.extend(batch_pred_boxes)
            all_target_boxes.extend(batch_target_boxes)
            
            # Store segmentation predictions and targets
            all_pred_masks.append(seg_output)
            all_target_masks.append(seg_targets)
    
    # Calculate average test loss
    test_loss /= len(test_loader)
    
    # Evaluate object detection on test set
    test_detection_metrics = {}
    if all_pred_boxes:
        test_detection_metrics = calculate_map(
            all_pred_boxes,
            all_target_boxes,
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                            0.95],
            class_names=class_names
        )
    
    # Evaluate segmentation on test set
    test_segmentation_metrics = {}
    if all_pred_masks and all_target_masks:
        pred_masks_cat = torch.cat(all_pred_masks, dim=0)
        target_masks_cat = torch.cat(all_target_masks, dim=0)
        
        test_segmentation_metrics = calculate_iou(
            pred_masks_cat, 
            target_masks_cat, 
            num_classes=len(class_names)
        )
    
    # Print test results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"mAP@.5: {test_detection_metrics.get('mAP@.5', 0):.4f}")
    print(f"mAP@.75: {test_detection_metrics.get('mAP@.75', 0):.4f}")
    print(f"mAP@[0.5:0.95]: {test_detection_metrics.get('mAP@[0.5:0.95]', 0):.4f}")
    print(f"mIoU: {test_segmentation_metrics.get('mIoU', 0):.4f}")

    # Print per-class IoU
    for cls_idx in range(1, len(class_names) + 1):
        cls_name = class_names[cls_idx - 1]
        cls_iou = test_segmentation_metrics.get(f"IoU_class_{cls_idx}", 0)
        print(f"IoU {cls_name}: {cls_iou:.4f}")

    # Load training history
    with open(root_dir / 'weights' / 'training_history.json', 'r') as f:
        history = json.load(f)

    # Plot training history
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['det_loss'], label='Detection Loss')
    plt.plot(history['seg_loss'], label='Segmentation Loss')
    plt.title('Task-Specific Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['mAP'], label='mAP@.5')
    plt.title('Detection Performance (mAP@.5)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history['mIoU'], label='mIoU')
    plt.title('Segmentation Performance (mIoU)')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(root_dir / 'results' / 'training_history.png')
    plt.show()

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
