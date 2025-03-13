from pathlib import Path
import json
from tqdm import tqdm
import torch
import torch.optim as optim
from utils import non_max_suppression, calculate_map, calculate_iou, \
    YOLOUNet, MultiTaskLoss
from dataloader import get_data_loaders


# Training function
def train_model(model, train_loader, val_loader, class_names, device,
                num_epochs=30, learning_rate=1e-4, weight_decay=1e-5,
                early_stopping_patience=5,
                checkpoint_dir=Path(__file__).parents[1] / "weights"):
    """
    Train the multi-task model.

    Args:
        model: The YOLOUNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_names: List of class names
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        early_stopping_patience: Number of epochs to wait for improvement
        before stopping
        checkpoint_dir: Directory to save model checkpoints

    Returns:
        Trained model and training history
    """
    model = model.to(device)

    # Compute class weights for segmentation loss
    # This helps address class imbalance
    class_weights = None
    if train_loader.dataset.class_weights is not None:
        class_weights = torch.tensor(
            train_loader.dataset.class_weights).to(device)

    # Initialize loss function and optimizer
    criterion = MultiTaskLoss(lambda_coord=5.0, lambda_noobj=0.5,
                              lambda_seg=1.0, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,
                                                     factor=0.1, verbose=True)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'det_loss': [],
        'seg_loss': [],
        'mAP': [],
        'mIoU': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_det_loss = 0.0
        train_seg_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, yolo_targets, seg_targets) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            yolo_targets = yolo_targets.to(device)
            seg_targets = seg_targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            yolo_output, seg_output = model(images)

            # Calculate loss
            loss, det_loss, seg_loss = criterion(
                (yolo_output, seg_output),
                (yolo_targets, seg_targets)
            )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            train_det_loss += det_loss.item()
            train_seg_loss += seg_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'det_loss': det_loss.item(),
                'seg_loss': seg_loss.item()
            })

        # Calculate average training losses
        train_loss /= len(train_loader)
        train_det_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_pred_boxes = []
        all_target_boxes = []
        all_pred_masks = []
        all_target_masks = []

        with torch.no_grad():
            for batch_idx, (images, yolo_targets, seg_targets) in enumerate(val_loader):
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

                val_loss += loss.item()

                # Process predictions for evaluation
                # Convert YOLO output to boxes
                batch_pred_boxes = []
                batch_target_boxes = []

                for i in range(images.size(0)):
                    pred_boxes = []
                    target_boxes = []

                    # Process each grid cell
                    for row in range(model.grid_size):
                        for col in range(model.grid_size):
                            # For box 1
                            pred_box1 = yolo_output[i, row, col, :5].clone()
                            confidence1 = pred_box1[4]

                            if confidence1 > 0.1:  # Minimum confidence
                                # Convert to absolute coordinates
                                pred_box1[0] = (pred_box1[0] + col
                                                ) / model.grid_size
                                pred_box1[1] = (pred_box1[1] + row
                                                ) / model.grid_size

                                # Get class
                                class_scores = yolo_output[i, row, col,
                                                           5 * model.num_boxes:
                                                           ].clone()
                                class_idx = torch.argmax(class_scores).item()

                                # Add to predictions
                                pred_boxes.append(
                                    torch.cat([pred_box1,
                                               torch.tensor([class_idx],
                                                            device=device)]))

                            # For box 2 if applicable
                            if model.num_boxes > 1:
                                pred_box2 = yolo_output[i, row, col, 5:10].clone()
                                confidence2 = pred_box2[4]

                                if confidence2 > 0.1:
                                    # Convert to absolute coordinates
                                    pred_box2[0] = (pred_box2[0] + col) / model.grid_size
                                    pred_box2[1] = (pred_box2[1] + row) / model.grid_size

                                    # Get class (same as box 1 in YOLOv1)
                                    class_scores = yolo_output[i, row, col, 5*model.num_boxes:].clone()
                                    class_idx = torch.argmax(class_scores).item()

                                    # Add to predictions [x, y, w, h, confidence, class]
                                    pred_boxes.append(torch.cat([pred_box2, torch.tensor([class_idx], device=device)]))

                            # Process target boxes
                            target_box1 = yolo_targets[i, row, col, :5].clone()
                            if target_box1[4] > 0:  # Has object
                                # Convert to absolute coordinates
                                target_box1[0] = (target_box1[0] + col) / model.grid_size
                                target_box1[1] = (target_box1[1] + row) / model.grid_size

                                # Get class
                                target_class_scores = yolo_targets[i, row, col, 5*model.num_boxes:].clone()
                                target_class_idx = torch.argmax(target_class_scores).item()

                                # Add to targets [x, y, w, h, class]
                                target_boxes.append(torch.cat([target_box1[:4], torch.tensor([target_class_idx], device=device)]))

                            # For target box 2 if applicable
                            if model.num_boxes > 1:
                                target_box2 = yolo_targets[i, row, col, 5:10].clone()
                                if target_box2[4] > 0:  # Has object
                                    # Convert to absolute coordinates
                                    target_box2[0] = (target_box2[0] + col) / model.grid_size
                                    target_box2[1] = (target_box2[1] + row) / model.grid_size

                                    # Get class
                                    target_class_scores = yolo_targets[i, row, col, 5*model.num_boxes:].clone()
                                    target_class_idx = torch.argmax(target_class_scores).item()

                                    # Add to targets [x, y, w, h, class]
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

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Evaluate object detection
        detection_metrics = {}
        if all_pred_boxes:
            detection_metrics = calculate_map(
                all_pred_boxes,
                all_target_boxes,
                iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                                0.9, 0.95],
                class_names=class_names
            )

        # Evaluate segmentation
        segmentation_metrics = {}
        if all_pred_masks and all_target_masks:
            # Concatenate masks across batches
            pred_masks_cat = torch.cat(all_pred_masks, dim=0)
            target_masks_cat = torch.cat(all_target_masks, dim=0)

            segmentation_metrics = calculate_iou(
                pred_masks_cat,
                target_masks_cat,
                num_classes=len(class_names)
            )

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Detection Loss: {train_det_loss:.4f}")
        print(f"Segmentation Loss: {train_seg_loss:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['det_loss'].append(train_det_loss)
        history['seg_loss'].append(train_seg_loss)
        history['mAP'].append(detection_metrics.get('mAP@.5', 0))
        history['mIoU'].append(segmentation_metrics.get('mIoU', 0))

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), checkpoint_dir / 'weights.pth')
            print("Model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}")

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    # Save training history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class_names = ["Car", "Pedestrian", "Bicyclist", "MotorcycleScooter",
                   "Truck_Bus"]

    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    grid_size = 7
    num_boxes = 19
    num_classes = 5

    # Dataset paths
    root_dir = Path(__file__).parents[1]

    train_loader, val_loader, _, _ = get_data_loaders(
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

    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )


if __name__ == "__main__":
    main()