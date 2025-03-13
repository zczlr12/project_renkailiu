import os
import csv
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch as th
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kagglehub


def download_dataset(img_dir):
    """Download and extract the dataset to the given directory."""
    source = kagglehub.dataset_download("carlolepelaars/camvid")
    print(f"Copying dataset from {source} to {img_dir}...")
    shutil.copytree(Path(source) / "CamVid", img_dir)
    # Remove caches
    shutil.rmtree(source)


def process_ground_truth(img_dir, annot_dir, classes):
    df = pd.read_csv(img_dir / "class_dict.csv")

    # Process each dataset split
    for split in ['train', 'test', 'val']:
        print(f"Processing {split} dataset...")
        files = sorted(os.listdir(img_dir / split))

        for image_file in files:
            mask_file = image_file.replace('.png', '_L.png')
            annot_file = f'{image_file[:-4]}.csv'
            mask_split = split + '_labels'

            # Load mask
            mask_path = img_dir / mask_split / mask_file
            csv_path = annot_dir / split / annot_file
            mask = Image.open(mask_path)

            # Extract bounding boxes for each object class
            with open(csv_path, 'w') as csv_file:
                csv_file.write("object,x,y,w,h")
                mask_np = np.array(mask)
                for obj in classes:
                    # Get mask for current object class
                    obj_id = df[df['name'] == obj]
                    obj_mask = ((mask_np[:, :, 0] == obj_id['r'].values[0]) &
                                (mask_np[:, :, 1] == obj_id['g'].values[0]) &
                                (mask_np[:, :, -1] == obj_id['b'].values[0])
                                ).astype(np.uint8)

                    # Get bounding boxes for current object class
                    stats = cv2.connectedComponentsWithStats(
                        obj_mask,
                        connectivity=4
                    )[2][1:]

                    # Add bounding boxes to XML file
                    for stat in stats:
                        x, y, w, h, area = stat
                        if area >= 100:
                            csv_file.write(f"\n{obj},{x + w/2},{y + h/2},{w},"
                                           f"{h}")


class CamVid(Dataset):
    """
    A custom Dataset for the CamVid data. An index number (starting from 0)
    and a color is assigned to each of the labels of the dataset.
    """
    # For object detection
    index2label = ["Car", "Pedestrian", "Bicyclist", "MotorcycleScooter",
                   "Truck_Bus"]

    label2index = {label: index for index, label in enumerate(index2label)}

    label_clrs = ["#400080", "#404000", "#0080c0", "#c000c0", "#c080c0"]

    # For semantic segmentation
    colour2index = {
        (64, 0, 128): 1,    # Car
        (64, 64, 0): 2,     # Pedestrian
        (0, 128, 192): 3,   # Bicyclist
        (192, 0, 192): 4,   # MotorcycleScooter
        (192, 128, 192): 5  # Truck_Bus
        # All other colours map to background (0)
    }

    def __init__(self, root_dir, split, transform, S=7, B=19, C=5):
        """ Initialize the Camvid Dataset object.

        :param root_dir: The root directory of the dataset.
        :param split: The split of the dataset.
        :param transform: The transform that are applied to the images (x)
        and their corresponding targets (y).
        :param target_size: The target size of the images.
        """

        assert split == 'train' or split == 'test' or split == 'val'

        self.img_dir = root_dir / "camvid" / split
        self.mask_dir = root_dir / "camvid" / f"{split}_labels"
        self.annot_dir = root_dir / "annotations" / split

        self.pseudonyms = [
            filename[:-4] for filename in os.listdir(self.annot_dir)
        ]

        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        """
        Return the total number of instances of the dataset.

        :return: total instances of the dataset
        """
        return len(self.pseudonyms)

    def __getitem__(self, idx):
        """
        Given an index number in range [0, dataset's length) , return the
        corresponding image, mask and bbox target. If transform is defined,
        the outputs are first transformed and then return by the function.

        :param idx: The given index number
        :return: The image, mask and bbox target
        """
        pid = self.pseudonyms[idx]
        img_path = self.img_dir / f'{pid}.png'
        annot_path = os.path.join(self.annot_dir, f'{pid}.csv')
        mask_path = self.mask_dir / f'{pid}_L.png'

        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("RGB"))

        # Convert RGB mask to class indices
        class_mask = self.convert_rgb_to_labels(mask)

        # Create empty YOLO target
        yolo_target = np.zeros((self.S, self.S, 5 * self.B + self.C))

        # Basic transformation
        img = transforms.ToTensor()(img)
        img = transforms.Resize((224, 224))(img)
        class_mask = th.from_numpy(class_mask).long()
        class_mask = F.interpolate(
            class_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224),
            mode='nearest'
        ).squeeze().long()

        # Fill YOLO target with bounding box data
        bboxes = []
        with open(annot_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)                    # Remove the header
            for row in csv_reader:
                bboxes.append([self.label2index[row[0]]] +
                              [float(row[i]) for i in range(1, 5)])

        img_h, img_w = 224, 224  # Assuming resized to 224x224
        for i, (id, x, y, w, h) in enumerate(bboxes):
            # Convert absolute coordinates to relative coordinates (0-1)
            center_x = x / img_w
            center_y = y / img_h
            width = w / img_w
            height = h / img_h

            # Determine which grid cell the center falls into
            grid_x = int(center_x * self.S)
            grid_y = int(center_y * self.S)

            # Adjust to relative coordinates within grid cell
            center_x_cell = center_x * self.S - grid_x
            center_y_cell = center_y * self.S - grid_y

            # Ensure grid indices are within bounds
            grid_x = min(self.S - 1, max(0, grid_x))
            grid_y = min(self.S - 1, max(0, grid_y))

            # First box format: [x, y, w, h, confidence]
            yolo_target[grid_y, grid_x, 5*i:5*i+4] = [
                center_x_cell, center_y_cell, width, height]
            yolo_target[grid_y, grid_x, 5*i+4] = 1  # Confidence
            # Class one-hot encoding starts at index 5*Bs
            yolo_target[grid_y, grid_x, 5*self.B + id] = 1

        yolo_target = th.from_numpy(yolo_target).float()

        # Create one-hot encoded segmentation target
        seg_target = F.one_hot(
            class_mask, num_classes=self.C+1).permute(2, 0, 1).float()

        return img, yolo_target, seg_target

    def convert_rgb_to_labels(self, rgb_mask):
        """
        Convert RGB mask to class labels.
        Args:
            rgb_mask: RGB mask image as numpy array of shape (H, W, 3)
        Returns:
            label_mask: Label mask as numpy array of shape (H, W)
        """
        height, width, _ = rgb_mask.shape
        label_mask = np.zeros((height, width), dtype=np.uint8)

        # Find the closest color for each pixel
        for (r, g, b), class_idx in self.colour2index.items():
            # Find pixels that match this color
            mask = (rgb_mask[:, :, 0] == r) & (rgb_mask[:, :, 1] == g
                                               ) & (rgb_mask[:, :, 2] == b)
            label_mask[mask] = class_idx

        return label_mask


def get_data_loaders(root_dir, batch_size=8, grid_size=7, num_boxes=19,
                     num_classes=5):
    """
    Create data loaders for train, validation, and test sets

    Args:
        root_dir: Path to CamVid dataset
        batch_size: Batch size for data loaders
        grid_size: Grid size for YOLO
        num_boxes: Number of bounding boxes per grid cell
        num_classes: Number of classes

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CamVid(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        S=grid_size,
        B=num_boxes,
        C=num_classes
    )

    val_dataset = CamVid(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        S=grid_size,
        B=num_boxes,
        C=num_classes
    )

    test_dataset = CamVid(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        S=grid_size,
        B=num_boxes,
        C=num_classes
    )

    # Calculate class weights for handling imbalance
    # This method counts the frequency of each class in the dataset
    print("Calculating class weights to handle imbalance...")
    class_counts = np.zeros(num_classes + 1)  # +1 for background

    for _, _, seg_target in train_dataset:
        labels = th.argmax(seg_target, dim=0).unique()
        for label in labels:
            if label < len(class_counts):
                class_counts[label.item()] += 1

    # Calculate class weights inversely proportional to frequency
    # The rarer the class, the higher the weight
    class_weights = np.sum(class_counts) / (class_counts * len(class_counts))
    class_weights[class_counts == 0] = 0  # Handle classes with no samples

    # Apply sigmoid normalization to avoid extreme weights
    class_weights = 1 / (1 + np.exp(-0.1 * (class_weights -
                                            np.median(class_weights))))

    # Save weights to dataset for use in loss function
    train_dataset.class_weights = th.from_numpy(class_weights).float()

    print(f"Class weights: {class_weights}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == '__main__':
    data_dir = Path(__file__).parents[1] / "data"
    img_dir = data_dir / "camvid"
    annot_dir = data_dir / "annotations"
    classes = ['Car', 'Pedestrian', 'Bicyclist', 'MotorcycleScooter',
               'Truck_Bus']

    if not img_dir.exists():
        download_dataset(img_dir)
    process_ground_truth(img_dir, annot_dir, classes)
