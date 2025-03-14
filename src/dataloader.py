import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from utils import CLASS_MAPPING, COLOUR_MAPPING


class CamVid(Dataset):
    def __init__(self, root_dir, split='train', img_size=(512, 512),
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'val', or 'test'
            img_size (tuple): Target size for resizing images
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.images_dir = os.path.join(root_dir, split)
        self.masks_dir = os.path.join(root_dir, split + '_labels')

        self.images = sorted([os.path.join(self.images_dir, img)
                              for img in os.listdir(self.images_dir)
                              if img.endswith('.png')])
        self.masks = sorted([os.path.join(self.masks_dir, mask)
                             for mask in os.listdir(self.masks_dir)
                             if mask.endswith('.png')])

        # Preprocess once to get bounding boxes and save them
        self.annotations = self._generate_annotations()

    def __len__(self):
        return len(self.images)

    def _generate_annotations(self):
        """Generate bounding boxes from segmentation masks."""
        annotations = []

        for mask_path in self.masks:
            mask = cv2.imread(mask_path)
            boxes = []
            classes = []

            for class_name, colour in COLOUR_MAPPING.items():
                # Create binary mask for this class
                binary_mask = np.all(mask == colour, axis=2).astype(np.uint8)

                # Find contours in the binary mask
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # Filter out small contours (noise)
                    if cv2.contourArea(contour) < 100:
                        continue

                    # Get bounding box coordinates
                    x, y, w, h = cv2.boundingRect(contour)

                    # Add to boxes and classes
                    boxes.append([x, y, x+w, y+h])
                    classes.append(CLASS_MAPPING[class_name])

            annotations.append({
                'boxes': boxes,
                'classes': classes
            })

        return annotations

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        original_size = img.size  # (width, height)

        # Load mask
        mask_path = self.masks[idx]
        mask = np.array(Image.open(mask_path).convert('RGB'))

        # Resize image and mask
        img = img.resize(self.img_size, Image.BILINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Convert mask to class indices
        seg_mask = np.zeros(self.img_size, dtype=np.int64)
        for class_name, color in COLOUR_MAPPING.items():
            class_idx = CLASS_MAPPING[class_name]
            seg_mask[np.all(mask == color, axis=2)] = class_idx

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])(img)

        # Get bounding boxes
        boxes = torch.tensor(self.annotations[idx]['boxes'],
                             dtype=torch.float32)
        classes = torch.tensor(self.annotations[idx]['classes'],
                               dtype=torch.int64)

        # Scale boxes to match resized image
        if len(boxes) > 0:
            scale_x = self.img_size[0] / original_size[0]
            scale_y = self.img_size[1] / original_size[1]

            boxes[:, 0] *= scale_x  # x_min
            boxes[:, 2] *= scale_x  # x_max
            boxes[:, 1] *= scale_y  # y_min
            boxes[:, 3] *= scale_y  # y_max

        # Prepare segmentation mask
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': classes,
            'seg_mask': seg_mask,
            'img_path': img_path,
            'original_size': original_size
        }

        return img, target


def get_data_loaders(root_dir, batch_size=4, img_size=(512, 512),
                     num_workers=4):
    """
    Create data loaders for training, validation, and testing.

    Args:
        root_dir (string): Directory with all the images.
        batch_size (int): Batch size for the data loaders.
        img_size (tuple): Target size for resizing images.
        num_workers (int): Number of workers for data loading.

    Returns:
        dict: Dictionary containing data loaders for train, val, and test.
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                               hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                              0.225])
    ])

    # No augmentation for validation and testing
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                              0.225])
    ])

    # Create datasets
    train_dataset = CamVid(root_dir, split='train', img_size=img_size,
                           transform=train_transform)
    val_dataset = CamVid(root_dir, split='val', img_size=img_size,
                         transform=val_transform)
    test_dataset = CamVid(root_dir, split='test', img_size=img_size,
                          transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn(batch):
    """Custom collate function to handle variable-size bounding boxes."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    images = torch.stack(images)

    return images, targets


# Get class weights for handling class imbalance
def get_class_weights(dataset):
    """Calculate class weights for segmentation."""
    class_counts = np.zeros(len(CLASS_MAPPING))
    total_pixels = 0

    for i in range(len(dataset)):
        _, target = dataset[i]
        seg_mask = target['seg_mask']
        for class_idx in range(len(CLASS_MAPPING)):
            class_counts[class_idx] += torch.sum(seg_mask == class_idx).item()
        total_pixels += seg_mask.numel()

    class_weights = np.ones_like(class_counts)
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            class_weights[i] = total_pixels / (len(CLASS_MAPPING) *
                                               class_counts[i])

    return torch.tensor(class_weights, dtype=torch.float32)
