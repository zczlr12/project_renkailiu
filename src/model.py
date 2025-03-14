import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block for YOLOv8 and UNet."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) activation as used in YOLOv8

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with bottleneck design for YOLOv8."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        bottleneck_channels = channels // 2

        self.conv1 = ConvBlock(channels, bottleneck_channels, kernel_size=1,
                               padding=0)
        self.conv2 = ConvBlock(bottleneck_channels, channels, kernel_size=3,
                               padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class DownSample(nn.Module):
    """Downsample block for encoder path."""
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3,
                              stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """Upsample block for decoder path."""
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class CSPLayer(nn.Module):
    """Cross Stage Partial Network (CSP) layer for YOLOv8."""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(CSPLayer, self).__init__()
        bottleneck_channels = out_channels // 2

        self.conv1 = ConvBlock(in_channels, bottleneck_channels, kernel_size=1,
                               padding=0)
        self.conv2 = ConvBlock(in_channels, bottleneck_channels, kernel_size=1,
                               padding=0)

        self.blocks = nn.Sequential(*[
            ResidualBlock(bottleneck_channels) for _ in range(num_blocks)
        ])

        self.conv3 = ConvBlock(bottleneck_channels * 2, out_channels,
                               kernel_size=1, padding=0)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.blocks(y1)

        y2 = self.conv2(x)

        y = torch.cat([y1, y2], dim=1)
        y = self.conv3(y)

        return y


class Encoder(nn.Module):
    """Encoder for YOLOv8-UNet architecture."""
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        # Initial conv
        self.conv1 = ConvBlock(in_channels, 32)

        # Encoder blocks (downsample + CSP)
        self.down1 = DownSample(32, 64)
        self.csp1 = CSPLayer(64, 64, num_blocks=1)

        self.down2 = DownSample(64, 128)
        self.csp2 = CSPLayer(128, 128, num_blocks=2)

        self.down3 = DownSample(128, 256)
        self.csp3 = CSPLayer(256, 256, num_blocks=3)

        self.down4 = DownSample(256, 512)
        self.csp4 = CSPLayer(512, 512, num_blocks=1)

    def forward(self, x):
        # Initial conv
        x0 = self.conv1(x)

        # Encoder path
        x1 = self.down1(x0)
        x1 = self.csp1(x1)

        x2 = self.down2(x1)
        x2 = self.csp2(x2)

        x3 = self.down3(x2)
        x3 = self.csp3(x3)

        x4 = self.down4(x3)
        x4 = self.csp4(x4)

        return x0, x1, x2, x3, x4


class Decoder(nn.Module):
    """UNet-style decoder for segmentation."""
    def __init__(self, num_classes=5):
        super(Decoder, self).__init__()

        # Upsampling path
        self.up1 = UpSample(512, 256)
        self.up2 = UpSample(256, 128)
        self.up3 = UpSample(128, 64)
        self.up4 = UpSample(64, 32)

        # Final segmentation head
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features

        # Decoder path
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # Segmentation head
        seg_output = self.seg_head(x)

        return seg_output


class DetectionHead(nn.Module):
    """YOLOv8-style detection head."""
    def __init__(self, num_classes=5, num_anchors=3):
        super(DetectionHead, self).__init__()

        # For each anchor: [x, y, w, h, objectness, num_classes]
        self.num_outputs = 5 + num_classes
        self.num_anchors = num_anchors

        # Detection heads at different scales
        self.head1 = nn.Conv2d(512, self.num_anchors * self.num_outputs,
                               kernel_size=1)
        self.head2 = nn.Conv2d(256, self.num_anchors * self.num_outputs,
                               kernel_size=1)
        self.head3 = nn.Conv2d(128, self.num_anchors * self.num_outputs,
                               kernel_size=1)

    def forward(self, features):
        _, _, x2, x3, x4 = features

        # Apply detection heads
        det1 = self.head1(x4)  # High level features (low resolution)
        det2 = self.head2(x3)  # Mid level features
        det3 = self.head3(x2)  # Low level features (high resolution)

        # Reshape outputs
        batch_size = det1.shape[0]

        det1 = det1.view(batch_size, self.num_anchors, self.num_outputs,
                         det1.shape[2], det1.shape[3])
        det1 = det1.permute(0, 1, 3, 4, 2).contiguous()

        det2 = det2.view(batch_size, self.num_anchors, self.num_outputs,
                         det2.shape[2], det2.shape[3])
        det2 = det2.permute(0, 1, 3, 4, 2).contiguous()

        det3 = det3.view(batch_size, self.num_anchors, self.num_outputs,
                         det3.shape[2], det3.shape[3])
        det3 = det3.permute(0, 1, 3, 4, 2).contiguous()

        return [det1, det2, det3]


class Model(nn.Module):
    """
    Combined YOLOv8 and UNet architecture for joint object detection and
    segmentation.
    """
    def __init__(self, in_channels=3, num_classes=5, num_anchors=3):
        super(Model, self).__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(num_classes)
        self.det_head = DetectionHead(num_classes, num_anchors)

        # Anchor boxes for each detection scale (normalized to [0,1])
        self.anchors = torch.tensor([
            [[0.10, 0.13], [0.16, 0.30], [0.33, 0.23]],  # Small objects
            [[0.30, 0.61], [0.62, 0.45], [0.42, 0.80]],  # Medium objects
            [[0.70, 0.72], [0.85, 0.90], [0.95, 0.45]]   # Large objects
        ], dtype=torch.float32)

        self.num_classes = num_classes

    def forward(self, x):
        features = self.encoder(x)

        # Segmentation branch
        seg_output = self.decoder(features)

        # Detection branch
        det_outputs = self.det_head(features)

        return {
            'detection': det_outputs,
            'segmentation': seg_output
        }

    def _box_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        # Get coordinates of intersection
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])

        # Calculate area of intersection
        intersect_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1,
                                                                   min=0)

        # Calculate area of union
        box1_area = (
            box1[..., 2] - box1[..., 0]
        ) * (
            box1[..., 3] - box1[..., 1]
        )
        box2_area = (
            box2[..., 2] - box2[..., 0]
        ) * (
            box2[..., 3] - box2[..., 1]
        )
        union_area = box1_area + box2_area - intersect_area

        # Calculate IoU
        iou = intersect_area / union_area

        return iou

    def detect(self, x, conf_threshold=0.25, nms_threshold=0.45):
        """
        Perform object detection inference.

        Args:
            x (torch.Tensor): Input image tensor
            conf_threshold (float): Confidence threshold
            nms_threshold (float): NMS IoU threshold

        Returns:
            list: List of detections for each image in batch
        """
        device = x.device
        batch_size = x.shape[0]
        self.anchors = self.anchors.to(device)

        # Forward pass
        outputs = self.forward(x)
        det_outputs = outputs['detection']

        # Process each image in the batch
        results = []
        for batch_idx in range(batch_size):
            # Hold all detections for this image
            image_boxes = []
            image_scores = []
            image_classes = []

            # Process each detection scale
            for scale_idx, detection in enumerate(det_outputs):
                # Get detection for this batch and scale
                det = detection[batch_idx]

                # Get grid dimensions
                grid_h, grid_w = det.shape[1:3]
                stride_h = x.shape[2] / grid_h
                stride_w = x.shape[3] / grid_w

                # Create grid offsets
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(grid_h, device=device),
                    torch.arange(grid_w, device=device),
                    indexing='ij'
                )

                # Reshape for broadcasting with detections
                grid_x = grid_x.unsqueeze(0).repeat(self.anchors.shape[1], 1,
                                                    1)
                grid_y = grid_y.unsqueeze(0).repeat(self.anchors.shape[1], 1,
                                                    1)
                anchors = self.anchors[scale_idx].unsqueeze(1).unsqueeze(1)\
                    .repeat(1, grid_h, grid_w, 1)

                # Extract predictions
                box_xy = torch.sigmoid(det[..., 0:2])  # Center x, y (0-1)
                box_wh = torch.exp(det[..., 2:4]) * anchors  # Width, height
                obj_conf = torch.sigmoid(det[..., 4:5])  # Confidence
                class_probs = torch.sigmoid(det[..., 5:])  # Probabilities

                # Calculate actual box coordinates
                box_xy = (box_xy + torch.stack([grid_x, grid_y], dim=3)) * \
                    torch.tensor([stride_w, stride_h], device=device)
                box_wh = box_wh * torch.tensor([stride_w, stride_h],
                                               device=device)

                # Convert to [x_min, y_min, x_max, y_max]
                box_x1y1 = box_xy - box_wh / 2
                box_x2y2 = box_xy + box_wh / 2
                boxes = torch.cat([box_x1y1, box_x2y2], dim=3)

                # Reshape boxes and scores for easier processing
                boxes = boxes.reshape(-1, 4)
                obj_conf = obj_conf.reshape(-1)
                class_probs = class_probs.reshape(-1, self.num_classes)

                # Get class with highest probability
                class_scores, class_ids = torch.max(class_probs, dim=1)
                scores = obj_conf * class_scores

                # Filter by confidence threshold
                mask = scores > conf_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
                filtered_classes = class_ids[mask]

                # Add to image results
                image_boxes.append(filtered_boxes)
                image_scores.append(filtered_scores)
                image_classes.append(filtered_classes)

            # Concatenate results from all scales
            if image_boxes and len(image_boxes[0]) > 0:
                image_boxes = torch.cat(image_boxes)
                image_scores = torch.cat(image_scores)
                image_classes = torch.cat(image_classes)

                # Apply NMS for each class
                final_boxes = []
                final_scores = []
                final_classes = []

                for cls in range(self.num_classes):
                    cls_mask = image_classes == cls
                    if not cls_mask.any():
                        continue

                    cls_boxes = image_boxes[cls_mask]
                    cls_scores = image_scores[cls_mask]

                    # Sort by score
                    _, indices = torch.sort(cls_scores, descending=True)
                    cls_boxes = cls_boxes[indices]
                    cls_scores = cls_scores[indices]

                    # Apply NMS
                    keep_indices = []
                    while cls_boxes.shape[0] > 0:
                        keep_indices.append(0)
                        if cls_boxes.shape[0] == 1:
                            break

                        # Calculate IoU with the top box
                        ious = torch.zeros(cls_boxes.shape[0] - 1,
                                           device=device)
                        for i in range(1, cls_boxes.shape[0]):
                            ious[i-1] = self._box_iou(cls_boxes[0:1],
                                                      cls_boxes[i:i+1])

                        # Remove boxes with IoU > threshold
                        mask = ious <= nms_threshold
                        cls_boxes = torch.cat([cls_boxes[0:1],
                                               cls_boxes[1:][mask]], dim=0)
                        cls_scores = torch.cat([cls_scores[0:1],
                                                cls_scores[1:][mask]], dim=0)

                    final_boxes.append(cls_boxes)
                    final_scores.append(cls_scores)
                    final_classes.extend([cls] * len(cls_boxes))

                if final_boxes:
                    final_boxes = torch.cat(final_boxes)
                    final_scores = torch.cat(final_scores)
                    final_classes = torch.tensor(final_classes, device=device)

                    results.append({
                        'boxes': final_boxes,
                        'scores': final_scores,
                        'classes': final_classes
                    })
                else:
                    results.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'classes': torch.zeros(0, dtype=torch.int64,
                                               device=device)
                    })
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'classes': torch.zeros(0, dtype=torch.int64, device=device)
                })

        return results

    def segment(self, x):
        """
        Perform semantic segmentation inference.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            torch.Tensor: Segmentation predictions
        """
        # Forward pass
        outputs = self.forward(x)
        seg_output = outputs['segmentation']

        # Apply softmax to get class probabilities
        seg_probs = F.softmax(seg_output, dim=1)

        # Get predicted class
        _, seg_preds = torch.max(seg_probs, dim=1)

        return seg_preds, seg_probs
