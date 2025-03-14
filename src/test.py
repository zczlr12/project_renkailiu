import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from dataloader import get_data_loaders
from model import Model
from utils import load_checkpoint, IDX_TO_CLASS, IDX_TO_COLOR


def test(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get data loader
    print("Initializing data loader...")
    data_loaders = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    data_loader = data_loaders['test']

    # Initialize model
    print("Initializing model...")
    model = Model(
        in_channels=3,
        num_classes=len(IDX_TO_CLASS),
        num_anchors=args.num_anchors
    ).to(device)

    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model, _, _, _ = load_checkpoint(model, None, args.checkpoint)
        print("Loaded checkpoint successfully")
    else:
        print(f"No checkpoint found at: {args.checkpoint}")
        return

    # Set model to evaluation mode
    model.eval()

    # Run inference
    print("Running inference...")
    results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            # Move data to device
            images = images.to(device)
            batch_size = images.shape[0]

            # Get detection and segmentation outputs
            det_results = model.detect(
                images,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold
            )
            seg_preds, _ = model.segment(images)

            # Process results
            for i in range(batch_size):
                # Get image info
                img_path = targets[i]['img_path']
                original_size = targets[i]['original_size']
                img_name = os.path.basename(img_path)

                # Get ground truth
                gt_boxes = targets[i]['boxes'].cpu()
                gt_labels = targets[i]['labels'].cpu()
                gt_seg_mask = targets[i]['seg_mask'].cpu()

                # Get predictions
                pred_boxes = det_results[i]['boxes'].cpu()
                pred_scores = det_results[i]['scores'].cpu()
                pred_classes = det_results[i]['classes'].cpu()
                pred_seg_mask = seg_preds[i].cpu()

                # Save results
                result = {
                    'img_path': img_path,
                    'img_name': img_name,
                    'original_size': original_size,
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'gt_seg_mask': gt_seg_mask,
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_classes': pred_classes,
                    'pred_seg_mask': pred_seg_mask
                }
                results.append(result)

                # Save segmentation prediction
                if args.save_outputs:
                    # Create color-coded segmentation mask
                    seg_mask_rgb = np.zeros((pred_seg_mask.shape[0],
                                             pred_seg_mask.shape[1], 3),
                                            dtype=np.uint8)
                    for class_idx, color in IDX_TO_COLOR.items():
                        seg_mask_rgb[pred_seg_mask == class_idx] = color

                    # Save segmentation mask
                    seg_mask_img = Image.fromarray(seg_mask_rgb)
                    seg_mask_img = seg_mask_img.resize(original_size,
                                                       Image.NEAREST)
                    seg_mask_img.save(os.path.join(args.output_dir,
                                                   f"seg_{img_name}"))

    # Save results for evaluation
    torch.save(results, os.path.join(args.output_dir, "results.pth"))
    print(f"Saved results to {os.path.join(args.output_dir, 'results.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a trained model")
    parser.add_argument("--data_dir", type=str, default="data/camvid",
                        help="Path to CamVid dataset")
    parser.add_argument("--checkpoint", type=str,
                        default="weights/best_checkpoint.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--img_size", type=tuple, default=(512, 512),
                        help="Image size")
    parser.add_argument("--num_anchors", type=int, default=3,
                        help="Number of anchors per scale")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--nms_threshold", type=float, default=0.45,
                        help="NMS threshold for detection")
    parser.add_argument("--save_outputs", action="store_true",
                        help="Save output segmentation masks and detections")

    args = parser.parse_args()
    test(args)
