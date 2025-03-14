import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloader import get_data_loaders, get_class_weights
from model import Model
from utils import save_checkpoint, load_checkpoint, focal_loss, dice_loss


def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get data loaders
    print("Initializing data loaders...")
    data_loaders = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    # Get class weights for segmentation loss
    train_dataset = data_loaders['train'].dataset
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights}")

    # Initialize model
    print("Initializing model...")
    model = Model(
        in_channels=3,
        num_classes=len(class_weights),
        num_anchors=args.num_anchors
    ).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Initialize best validation loss and starting epoch
    best_val_loss = float('inf')
    start_epoch = 0

    # Load checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            model, optimizer, start_epoch, best_val_loss = load_checkpoint(
                model, optimizer, args.resume
            )
            print(f"Loaded checkpoint (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at: {args.resume}")

    # Initialize loss functions
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
    detection_criterion = focal_loss  # Using focal loss for detection
    segmentation_dice_loss = dice_loss  # Using dice loss for segmentation

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_det_loss = 0.0
        train_seg_loss = 0.0

        train_loop = tqdm(data_loaders['train'],
                          desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (images, targets) in enumerate(train_loop):
            # Move data to device
            images = images.to(device)
            batch_size = images.shape[0]

            # Prepare targets
            det_targets = []
            seg_targets = []
            for target in targets:
                det_target = {
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                }
                det_targets.append(det_target)
                seg_targets.append(target['seg_mask'].to(device))

            seg_targets = torch.stack(seg_targets).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Get detection and segmentation outputs
            det_outputs = outputs['detection']
            seg_output = outputs['segmentation']

            # Calculate detection loss
            det_loss = 0.0
            for i in range(batch_size):
                # Calculate loss for each detection scale
                for scale_idx, scale_output in enumerate(det_outputs):
                    # Skip if no ground truth boxes
                    if det_targets[i]['boxes'].shape[0] == 0:
                        continue

                    # Get predictions for this image
                    preds = scale_output[i]

                    # Calculate focal loss for objectness
                    obj_preds = preds[..., 4]

                    # Create ground truth for objectness
                    obj_targets = torch.zeros_like(obj_preds)

                    # Calculate detection loss
                    scale_det_loss = detection_criterion(obj_preds,
                                                         obj_targets)
                    det_loss += scale_det_loss

            # Normalize detection loss by batch size
            det_loss = det_loss / batch_size if batch_size > 0 else 0.0

            # Calculate segmentation loss (cross-entropy + dice)
            seg_ce_loss = seg_criterion(seg_output, seg_targets)
            seg_dice = segmentation_dice_loss(seg_output,
                                              seg_targets.unsqueeze(1))
            seg_loss = seg_ce_loss + seg_dice

            # Combine losses
            loss = args.det_weight * det_loss + args.seg_weight * seg_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            train_det_loss += det_loss.item()
            train_seg_loss += seg_loss.item()

            # Update progress bar
            train_loop.set_postfix({
                'loss': f"{loss.item():.4f}",
                'det_loss': f"{det_loss.item():.4f}",
                'seg_loss': f"{seg_loss.item():.4f}"
            })

        # Calculate average losses for the epoch
        train_loss /= len(data_loaders['train'])
        train_det_loss /= len(data_loaders['train'])
        train_seg_loss /= len(data_loaders['train'])

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_det_loss = 0.0
        val_seg_loss = 0.0

        val_loop = tqdm(data_loaders['val'],
                        desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loop):
                # Move data to device
                images = images.to(device)
                batch_size = images.shape[0]

                # Prepare targets
                det_targets = []
                seg_targets = []
                for target in targets:
                    det_target = {
                        'boxes': target['boxes'].to(device),
                        'labels': target['labels'].to(device)
                    }
                    det_targets.append(det_target)
                    seg_targets.append(target['seg_mask'].to(device))

                seg_targets = torch.stack(seg_targets).to(device)

                # Forward pass
                outputs = model(images)

                # Get detection and segmentation outputs
                det_outputs = outputs['detection']
                seg_output = outputs['segmentation']

                # Calculate detection loss
                det_loss = 0.0
                for i in range(batch_size):
                    # Calculate loss for each detection scale
                    for scale_idx, scale_output in enumerate(det_outputs):
                        # Skip if no ground truth boxes
                        if det_targets[i]['boxes'].shape[0] == 0:
                            continue

                        # Get predictions for this image
                        preds = scale_output[i]

                        # Calculate focal loss for objectness
                        obj_preds = preds[..., 4]

                        # Create ground truth for objectness
                        obj_targets = torch.zeros_like(obj_preds)

                        # Calculate detection loss
                        scale_det_loss = detection_criterion(obj_preds,
                                                             obj_targets)
                        det_loss += scale_det_loss

                # Normalize detection loss by batch size
                det_loss = det_loss / batch_size if batch_size > 0 else 0.0

                # Calculate segmentation loss (cross-entropy + dice)
                seg_ce_loss = seg_criterion(seg_output, seg_targets)
                seg_dice = segmentation_dice_loss(seg_output,
                                                  seg_targets.unsqueeze(1))
                seg_loss = seg_ce_loss + seg_dice

                # Combine losses
                loss = args.det_weight * det_loss + args.seg_weight * seg_loss

                # Update statistics
                val_loss += loss.item()
                val_det_loss += det_loss.item()
                val_seg_loss += seg_loss.item()

                # Update progress bar
                val_loop.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'det_loss': f"{det_loss.item():.4f}",
                    'seg_loss': f"{seg_loss.item():.4f}"
                })

        # Calculate average losses for the epoch
        val_loss /= len(data_loaders['val'])
        val_det_loss /= len(data_loaders['val'])
        val_seg_loss /= len(data_loaders['val'])

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (Det: {train_det_loss:.4f}, Seg"
              f": {train_seg_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Det: {val_det_loss:.4f}, Seg: "
              f"{val_seg_loss:.4f})")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            best_val_loss,
            os.path.join(args.checkpoint_dir,
                         f"checkpoint_epoch_{epoch+1}.pth")
        )

        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                best_val_loss,
                os.path.join(args.checkpoint_dir, "best_checkpoint.pth")
            )

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-task model for"
                                     " object detection and segmentation")
    parser.add_argument("--data_dir", type=str, default="data/camvid",
                        help="Path to CamVid dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="weights",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--img_size", type=tuple, default=(512, 512),
                        help="Image size")
    parser.add_argument("--num_anchors", type=int, default=3,
                        help="Number of anchors per scale")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--det_weight", type=float, default=1.0,
                        help="Weight for detection loss")
    parser.add_argument("--seg_weight", type=float, default=1.0,
                        help="Weight for segmentation loss")

    args = parser.parse_args()
    train(args)
