import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

# Define the class mapping for CamVid
CLASS_MAPPING = {
    'Car': 0,
    'Pedestrian': 1,
    'Bicyclist': 2,
    'MotorcycleScooter': 3,
    'Truck_Bus': 4
}

# Reverse class mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

# Define RGB color mapping for visualization
COLOUR_MAPPING = {
    'Car': (64, 0, 128),
    'Pedestrian': (64, 64, 0),
    'Bicyclist': (0, 128, 192),
    'MotorcycleScooter': (192, 0, 192),
    'Truck_Bus': (192, 128, 192)
}

# Reverse color mapping
IDX_TO_COLOR = {CLASS_MAPPING[k]: COLOUR_MAPPING[k] for k in COLOUR_MAPPING}


def convert_to_yolo_format(boxes, img_size):
    """
    Convert bounding boxes from [x_min, y_min, x_max, y_max] to YOLO format
    [x_center, y_center, width, height].
    All values are normalized to [0, 1].
    """
    width, height = img_size

    # Convert to numpy for easier manipulation
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    yolo_boxes = np.zeros_like(boxes)

    # Convert to center format
    yolo_boxes[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / width  # x_center
    yolo_boxes[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / height  # y_center
    yolo_boxes[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width  # width
    yolo_boxes[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height  # height

    return torch.tensor(yolo_boxes, dtype=torch.float32)


def convert_from_yolo_format(boxes, img_size):
    """
    Convert bounding boxes from YOLO format [x_center, y_center, width, height
    ] to [x_min, y_min, x_max, y_max].
    All values are denormalized to original image size.
    """
    width, height = img_size

    # Convert to numpy for easier manipulation
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    regular_boxes = np.zeros_like(boxes)

    # Convert to corner format
    regular_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width  # x_min
    regular_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height  # y_min
    regular_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width  # x_max
    regular_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height  # y_max

    return torch.tensor(regular_boxes, dtype=torch.float32)


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply non-maximum suppression to remove overlapping boxes.

    Args:
        boxes (torch.Tensor): Bounding boxes in format [x_min, y_min, x_max,
        y_max]
        scores (torch.Tensor): Confidence scores for each box
        threshold (float): IoU threshold for NMS

    Returns:
        torch.Tensor: Indices of the boxes to keep
    """
    # Convert to numpy for easier manipulation
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    # If no boxes, return empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by the bottom-right y-coordinate
    idxs = np.argsort(scores)

    # Keep looping while some indexes still remain in the idxs list
    while len(idxs) > 0:
        # Grab the last index in the idxs list and add the index
        # value to the list of picked indexes
        i = idxs[0]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the
        # end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[1:]]

        # Delete all indexes from the index list that have overlap greater
        # than the threshold
        idxs = np.delete(
            idxs, np.concatenate(([0],
                                  np.where(overlap > threshold)[0] + 1)))

    return pick


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in format [x_min, y_min, x_max, y_max].

    Args:
        box1, box2 (torch.Tensor or numpy.ndarray): Bounding boxes

    Returns:
        float: IoU value
    """
    # Convert to numpy if tensor
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()

    # Get the coordinates of bounding boxes
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Get the coordinates of the intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate intersection area
    width_inter = max(0, x_max_inter - x_min_inter + 1)
    height_inter = max(0, y_max_inter - y_min_inter + 1)
    area_inter = width_inter * height_inter

    # Calculate union area
    area1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
    area2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)
    area_union = area1 + area2 - area_inter

    # Calculate IoU
    iou = area_inter / area_union

    return iou


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance in detection.

    Args:
        pred (torch.Tensor): Predicted values
        target (torch.Tensor): Target values
        alpha (float): Weighting factor for rare classes
        gamma (float): Focusing parameter to down-weight easy examples

    Returns:
        torch.Tensor: Focal loss value
    """
    bce_loss = F.binary_cross_entropy_with_logits(pred, target,
                                                  reduction='none')
    pt = torch.exp(-bce_loss)  # prob of correct class
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss

    return focal_loss.mean()


def dice_loss(pred, target, smooth=1.0):
    """
    Dice Loss for segmentation tasks.

    Args:
        pred (torch.Tensor): Predicted probabilities
        target (torch.Tensor): Target one-hot encoding
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        torch.Tensor: Dice loss value
    """
    pred = torch.sigmoid(pred)
    print(pred.shape)
    print(target.shape)

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    print(pred_flat.shape)
    print(target_flat.shape)

    # Calculate Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        return focal_loss(pred, target, self.alpha, self.gamma)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        return dice_loss(pred, target, self.smooth)


class ComboLoss(nn.Module):
    """
    Combined loss function for both detection and segmentation tasks.
    """
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0, det_weight=1.0,
                 seg_weight=1.0):
        super(ComboLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss(smooth)
        self.det_weight = det_weight
        self.seg_weight = seg_weight

    def forward(self, det_pred, det_target, seg_pred, seg_target):
        det_loss = self.focal_loss(det_pred, det_target)
        seg_loss = self.dice_loss(seg_pred, seg_target)

        return self.det_weight * det_loss + self.seg_weight * seg_loss


def save_checkpoint(model, optimizer, epoch, best_val_loss, filepath):
    """
    Save model checkpoint.

    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        best_val_loss (float): Best validation loss
        filepath (str): Path to save the checkpoint
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.

    Args:
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        filepath (str): Path to the checkpoint file

    Returns:
        tuple: (model, optimizer, epoch, best_val_loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, epoch, best_val_loss


def create_class_weight_map(dataset):
    """
    Create class weight map for handling class imbalance in segmentation.

    Args:
        dataset: Dataset containing the segmentation masks

    Returns:
        torch.Tensor: Weight for each class
    """
    class_counts = Counter()
    total_pixels = 0

    # Count the occurrences of each class
    for i in range(len(dataset)):
        _, target = dataset[i]
        mask = target['seg_mask']
        unique_values, counts = torch.unique(mask, return_counts=True)

        for val, count in zip(unique_values, counts):
            class_counts[val.item()] += count.item()

        total_pixels += mask.numel()

    # Create weight map
    num_classes = len(CLASS_MAPPING)
    weights = torch.ones(num_classes, dtype=torch.float32)

    for cls in range(num_classes):
        if cls in class_counts and class_counts[cls] > 0:
            weights[cls] = 1.0 / (class_counts[cls] / total_pixels)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights
