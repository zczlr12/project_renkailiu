import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


# UNet components
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Combined YOLOv1-UNet Multi-Task Model
class YOLOUNet(nn.Module):
    def __init__(self, grid_size=19, num_boxes=2, num_classes=5):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Base encoder - shared between YOLO and UNet
        backbone = models.resnet34(pretrained=True)
        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )  # 64 channels, 1/2 resolution
        self.pool = backbone.maxpool
        self.encoder2 = backbone.layer1  # 64 channels, 1/4 resolution
        self.encoder3 = backbone.layer2  # 128 channels, 1/8 resolution
        self.encoder4 = backbone.layer3  # 256 channels, 1/16 resolution
        self.encoder5 = backbone.layer4  # 512 channels, 1/32 resolution

        # YOLO Detection Pathway
        self.yolo_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        # Detection head
        self.yolo_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * (224 // 32) * (224 // 32), 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * (5 * num_boxes +
                                                     num_classes))
        )

        # UNet Segmentation Pathway - Decoder
        self.decoder1 = DoubleConv(512 + 256, 256)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.decoder3 = DoubleConv(128 + 64, 64)
        self.decoder4 = DoubleConv(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # Final segmentation layer
        self.seg_output = nn.Conv2d(32, num_classes + 1, kernel_size=1)

    def forward(self, x):
        # Encoder path - shared
        e1 = self.encoder1(x)  # 64 channels, 1/2 resolution
        e1_pool = self.pool(e1)  # 64 channels, 1/4 resolution
        e2 = self.encoder2(e1_pool)  # 64 channels, 1/4 resolution
        e3 = self.encoder3(e2)  # 128 channels, 1/8 resolution
        e4 = self.encoder4(e3)  # 256 channels, 1/16 resolution
        e5 = self.encoder5(e4)  # 512 channels, 1/32 resolution

        # YOLO Detection Path
        yolo_feat = self.yolo_conv(e5)
        yolo_output = self.yolo_head(yolo_feat)
        yolo_output = yolo_output.view(-1, self.grid_size, self.grid_size,
                                       5*self.num_boxes + self.num_classes)

        # UNet Segmentation Path
        d1 = self.upconv1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.decoder4(d4)

        d5 = self.upconv5(d4)
        seg_output = self.seg_output(d5)

        return yolo_output, seg_output


# Custom Loss Function for Multi-Task Learning
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, lambda_seg=1.0,
                 class_weights=None):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_seg = lambda_seg
        self.mse_loss = nn.MSELoss(reduction='sum')

        if class_weights is not None:
            self.seg_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.seg_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        yolo_pred, seg_pred = predictions
        yolo_target, seg_target = targets

        # Detection loss - YOLOv1 style
        # Reshape predictions and targets
        pred_boxes = yolo_pred[..., :10].contiguous()  # [batch, S, S, B*5]
        pred_classes = yolo_pred[..., 10:].contiguous()  # [batch, S, S, C]

        target_boxes = yolo_target[..., :10].contiguous()  # [batch, S, S, B*5]
        target_classes = yolo_target[..., 10:].contiguous()  # [batch, S, S, C]

        # Object mask (cells with objects)
        obj_mask = target_boxes[..., 4] > 0  # [batch, S, S]

        # Coordinate loss for cells with objects
        box1_pred = pred_boxes[..., :5].contiguous()
        box1_target = target_boxes[..., :5].contiguous()

        # Only consider cells with objects for coordinate and width/height loss
        box1_pred_xy = box1_pred[obj_mask][..., :2]
        box1_target_xy = box1_target[obj_mask][..., :2]

        box1_pred_wh = torch.sqrt(box1_pred[obj_mask][..., 2:4] + 1e-6)
        box1_target_wh = torch.sqrt(box1_target[obj_mask][..., 2:4] + 1e-6)

        # Coordinate loss
        xy_loss = self.mse_loss(box1_pred_xy, box1_target_xy)
        wh_loss = self.mse_loss(box1_pred_wh, box1_target_wh)

        # Confidence loss - for cells with and without objects
        # For cells with objects
        obj_confidence_loss = self.mse_loss(
            box1_pred[obj_mask][..., 4],
            box1_target[obj_mask][..., 4]
        )

        # For cells without objects
        noobj_mask = ~obj_mask
        noobj_confidence_loss = self.mse_loss(
            box1_pred[noobj_mask][..., 4],
            box1_target[noobj_mask][..., 4]
        )

        # Class loss - only for cells with objects
        class_loss = self.mse_loss(
            pred_classes[obj_mask],
            target_classes[obj_mask]
        )

        # Handle second box if available
        if box1_pred.size(-1) > 5:
            box2_pred = pred_boxes[..., 5:10].contiguous()
            box2_target = target_boxes[..., 5:10].contiguous()

            # Box 2 losses - similar to box 1
            box2_obj_mask = box2_target[..., 4] > 0

            if torch.sum(box2_obj_mask) > 0:
                box2_pred_xy = box2_pred[box2_obj_mask][..., :2]
                box2_target_xy = box2_target[box2_obj_mask][..., :2]

                box2_pred_wh = torch.sqrt(
                    box2_pred[box2_obj_mask][..., 2:4] + 1e-6)
                box2_target_wh = torch.sqrt(
                    box2_target[box2_obj_mask][..., 2:4] + 1e-6)

                # Add box 2 losses
                xy_loss += self.mse_loss(box2_pred_xy, box2_target_xy)
                wh_loss += self.mse_loss(box2_pred_wh, box2_target_wh)

                # Confidence loss for box 2
                obj_confidence_loss += self.mse_loss(
                    box2_pred[box2_obj_mask][..., 4],
                    box2_target[box2_obj_mask][..., 4]
                )

                box2_noobj_mask = ~box2_obj_mask
                noobj_confidence_loss += self.mse_loss(
                    box2_pred[box2_noobj_mask][..., 4],
                    box2_target[box2_noobj_mask][..., 4]
                )

        # Total YOLO loss
        det_loss = (
            self.lambda_coord * (xy_loss + wh_loss) +
            obj_confidence_loss +
            self.lambda_noobj * noobj_confidence_loss +
            class_loss
        )

        # Segmentation loss
        # Reshape for CrossEntropyLoss (expects [B, C, H, W])
        seg_pred = seg_pred.permute(0, 3, 1, 2)  # Convert to [B, C, H, W]
        seg_loss = self.lambda_seg * self.seg_loss(seg_pred,
                                                   seg_target.argmax(dim=1))

        # Total loss
        total_loss = det_loss + seg_loss

        return total_loss, det_loss, seg_loss


# Non-Maximum Suppression function for post-processing
def non_max_suppression(boxes, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression on the bounding boxes.

    Args:
        boxes: Tensor of shape [N, 6] containing bounding boxes and confidence.
               Each row is [x, y, w, h, confidence, class].
        confidence_threshold: Minimum confidence required to keep a box.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Tensor of shape [M, 6] containing the kept boxes after NMS.
    """
    if len(boxes) == 0:
        return []

    # Filter out boxes with low confidence
    keep = boxes[:, 4] > confidence_threshold
    boxes = boxes[keep]

    if len(boxes) == 0:
        return []

    # Get coordinates and compute areas
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    # Convert to corners format (x1, y1, x2, y2)
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2

    # Create a new array in corners format
    corners_format = torch.stack([x1, y1, x2, y2, boxes[:, 4], boxes[:, 5]],
                                 dim=1)

    # Sort boxes by confidence
    scores = boxes[:, 4]
    _, order = scores.sort(0, descending=True)
    corners_format = corners_format[order]

    keep = []
    while corners_format.size(0) > 0:
        # Pick the box with highest confidence and keep it
        keep.append(corners_format[0])

        # If only one box is left, break
        if len(corners_format) == 1:
            break

        # Compute IoU with the rest of the boxes
        current_box = corners_format[0, :4]
        rest_boxes = corners_format[1:, :4]

        # Calculate intersection coordinates
        xx1 = torch.max(current_box[0], rest_boxes[:, 0])
        yy1 = torch.max(current_box[1], rest_boxes[:, 1])
        xx2 = torch.min(current_box[2], rest_boxes[:, 2])
        yy2 = torch.min(current_box[3], rest_boxes[:, 3])

        # Calculate intersection area
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        intersection = w * h

        # Calculate area of boxes
        current_box_area = (current_box[2] - current_box[0]) * (
            current_box[3] - current_box[1])
        rest_boxes_area = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (
            rest_boxes[:, 3] - rest_boxes[:, 1])

        # Calculate union area
        union = current_box_area + rest_boxes_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)

        # Keep boxes with IoU less than threshold
        mask = iou < iou_threshold
        corners_format = corners_format[1:][mask]

    return torch.stack(keep) if keep else []


# Evaluation functions
def calculate_map(preds, targets, iou_thresholds=[0.5], class_names=None):
    """
    Calculate mAP for object detection.

    Args:
        preds: List of predicted boxes [x, y, w, h, confidence, class]
        targets: List of ground truth boxes [x, y, w, h, class]
        iou_thresholds: List of IoU thresholds for evaluation
        class_names: Optional list of class names for reporting

    Returns:
        Dictionary with mAP results
    """
    results = {}

    # If no predictions or targets, return empty results
    if not preds or not targets:
        return {"mAP": 0}

    # Group by class
    class_preds = {}
    class_targets = {}

    for box in preds:
        cls = int(box[5])
        if cls not in class_preds:
            class_preds[cls] = []
        class_preds[cls].append(box)

    for box in targets:
        cls = int(box[4])
        if cls not in class_targets:
            class_targets[cls] = []
        class_targets[cls].append(box)

    # Calculate AP for each class and IoU threshold
    aps = {}

    for iou_threshold in iou_thresholds:
        aps[iou_threshold] = []

        for cls in range(len(class_names) if class_names else max(
                         class_preds.keys()) + 1):
            if cls not in class_preds or cls not in class_targets:
                aps[iou_threshold].append(0)
                continue

            # Sort predictions by confidence
            sorted_preds = sorted(class_preds[cls], key=lambda x: x[4],
                                  reverse=True)

            # Create arrays for precision-recall calculation
            TP = torch.zeros(len(sorted_preds))
            FP = torch.zeros(len(sorted_preds))

            gt_boxes = class_targets[cls].copy()

            # Check each prediction
            for i, pred_box in enumerate(sorted_preds):
                best_iou = 0
                best_idx = -1

                # Convert to corner format
                pred_x1 = pred_box[0] - pred_box[2]/2
                pred_y1 = pred_box[1] - pred_box[3]/2
                pred_x2 = pred_box[0] + pred_box[2]/2
                pred_y2 = pred_box[1] + pred_box[3]/2

                # Check against all ground truth boxes
                for j, gt_box in enumerate(gt_boxes):
                    gt_x1 = gt_box[0] - gt_box[2]/2
                    gt_y1 = gt_box[1] - gt_box[3]/2
                    gt_x2 = gt_box[0] + gt_box[2]/2
                    gt_y2 = gt_box[1] + gt_box[3]/2

                    # Calculate IoU
                    x1 = max(pred_x1, gt_x1)
                    y1 = max(pred_y1, gt_y1)
                    x2 = min(pred_x2, gt_x2)
                    y2 = min(pred_y2, gt_y2)

                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)

                    intersection = w * h
                    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                    union = pred_area + gt_area - intersection

                    iou = intersection / union

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                # Check if the best match exceeds the IoU threshold
                if best_iou >= iou_threshold and best_idx >= 0:
                    TP[i] = 1
                    # Remove the matched ground truth box
                    gt_boxes.pop(best_idx)
                else:
                    FP[i] = 1

            # Calculate precision and recall
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)

            recalls = TP_cumsum / (len(class_targets[cls]) + 1e-6)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

            # Add sentinel values to start
            precisions = torch.cat([torch.tensor([1]), precisions])
            recalls = torch.cat([torch.tensor([0]), recalls])

            # Calculate average precision
            # Using 11-point interpolation method
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if torch.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = torch.max(precisions[recalls >= t])
                ap += p / 11

            aps[iou_threshold].append(float(ap))

    # Calculate mAP for each IoU threshold
    results["mAP@.5"] = sum(aps[0.5]) / len(aps[0.5]) if 0.5 in aps else 0

    # Calculate mAP@[0.5:0.95] (COCO metric)
    if len(iou_thresholds) > 1:
        all_aps = []
        for iou_threshold in iou_thresholds:
            all_aps.extend(aps[iou_threshold])
        results["mAP@[0.5:0.95]"] = sum(all_aps) / len(all_aps) if all_aps else 0

    # Per-class AP at IoU=0.5
    if class_names and 0.5 in aps:
        for i, ap in enumerate(aps[0.5]):
            if i < len(class_names):
                results[f"AP@.5_{class_names[i]}"] = ap

    return results


def calculate_iou(pred_mask, target_mask, num_classes):
    """
    Calculate IoU for segmentation.

    Args:
        pred_mask: Predicted segmentation mask [B, C, H, W]
        target_mask: Target segmentation mask [B, C, H, W]
        num_classes: Number of classes

    Returns:
        Dictionary with IoU results
    """
    results = {}

    # Convert to class indices
    pred_mask = torch.argmax(pred_mask, dim=1)  # [B, H, W]
    target_mask = torch.argmax(target_mask, dim=1)  # [B, H, W]

    # Calculate IoU for each class
    class_ious = []

    for cls in range(1, num_classes + 1):  # Skip background class 0
        # Create binary masks for current class
        pred_binary = (pred_mask == cls).float()
        target_binary = (target_mask == cls).float()

        # Calculate intersection and union
        intersection = torch.sum(pred_binary * target_binary, dim=(1, 2))  # [B]
        union = torch.sum(pred_binary, dim=(1, 2)) + torch.sum(target_binary, dim=(1, 2)) - intersection  # [B]

        # Calculate IoU
        iou = (intersection + 1e-6) / (union + 1e-6)  # [B]

        # Average over batch
        mean_iou = torch.mean(iou).item()
        class_ious.append(mean_iou)

        # Store class IoU
        results[f"IoU_class_{cls}"] = mean_iou

    # Mean IoU over all classes
    results["mIoU"] = sum(class_ious) / len(class_ious) if class_ious else 0

    return results
