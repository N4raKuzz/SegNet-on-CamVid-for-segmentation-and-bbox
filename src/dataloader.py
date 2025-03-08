import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Mapping for training (32 classes) using the CSV info
TRAIN_CLASS_COLORS = {
    (64, 128, 64): 0,      # Animal
    (192, 0, 128): 1,      # Archway
    (0, 128, 192): 2,      # Bicyclist
    (0, 128, 64): 3,       # Bridge
    (128, 0, 0): 4,        # Building
    (64, 0, 128): 5,       # Car
    (64, 0, 192): 6,       # CartLuggagePram
    (192, 128, 64): 7,     # Child
    (192, 192, 128): 8,    # Column_Pole
    (64, 64, 128): 9,      # Fence
    (128, 0, 192): 10,     # LaneMkgsDriv
    (192, 0, 64): 11,      # LaneMkgsNonDriv
    (128, 128, 64): 12,    # Misc_Text
    (192, 0, 192): 13,     # MotorcycleScooter
    (128, 64, 64): 14,     # OtherMoving
    (64, 192, 128): 15,    # ParkingBlock
    (64, 64, 0): 16,       # Pedestrian
    (128, 64, 128): 17,    # Road
    (128, 128, 192): 18,   # RoadShoulder
    (0, 0, 192): 19,       # Sidewalk
    (192, 128, 128): 20,   # SignSymbol
    (128, 128, 128): 21,   # Sky
    (64, 128, 192): 22,    # SUVPickupTruck
    (0, 0, 64): 23,        # TrafficCone
    (0, 64, 64): 24,       # TrafficLight
    (192, 64, 128): 25,    # Train
    (128, 128, 0): 26,     # Tree
    (192, 128, 192): 27,   # Truck_Bus
    (64, 0, 64): 28,       # Tunnel
    (192, 192, 0): 29,     # VegetationMisc
    (0, 0, 0): 30,         # Void
    (64, 192, 0): 31       # Wall
}

# Mapping for testing/validation (5 target classes)
TEST_CLASS_COLORS = {
    (0, 128, 192): 1,      # Bicyclist
    (64, 0, 128): 2,       # Car
    (64, 64, 0): 3,        # Pedestrian
    (192, 0, 192): 4,      # MotorcycleScooter
    (192, 128, 192): 5     # Truck_Bus
}

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', mode='segmentation', transform=None):
        """
        Directory structure:
            data/
            └── CamVid/
                ├── train/
                │   ├── images/ -> image files (*.png)
                │   └── masks/  -> segmentation mask images (*_L.png)
                ├── val/
                │   ├── images/
                │   └── masks/
                └── test/
                    ├── images/
                    └── masks/
            annotations/
                └── train_annotations.json
                └── val_annotations.json
                └── test_annotations.json

        :param root_dir: Root directory for the CamVid dataset.
        :param split: One of 'train', 'val', or 'test'.
        :param mode: 'segmentation' or 'det'.
        :param transform: Transformations to be applied to the image.
        :param class_mapping: Dictionary mapping RGB tuples to integer class IDs.
                              If not provided, defaults to TEST_CLASS_COLORS.
        """
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform
        # Use provided mapping or default to testing mapping
        self.class_mapping = TEST_CLASS_COLORS if self.split != 'train' else TRAIN_CLASS_COLORS

        self.image_dir = os.path.join(root_dir, 'CamVid', split, 'images')
        self.mask_dir = os.path.join(root_dir, 'CamVid', split, 'masks')
        self.annotation_file = os.path.join(root_dir, 'annotations', f'{split}_annotations.json')
        
        # Sorted list of image paths
        self.image_paths = sorted([
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith('.png')
        ])
        
        # Load bbox annotations if needed
        if self.mode == 'det':
            with open(self.annotation_file, 'r') as f:
                annotation_list = json.load(f)
                self.annotations = {entry['filename']: entry for entry in annotation_list}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'segmentation':
            mask_path = os.path.join(
                self.mask_dir,
                filename.replace('.png', '_L.png')
            )
            mask = Image.open(mask_path).convert('RGB')
            mask = self._encode_segmentation(mask)

            return image, mask

        elif self.mode == 'det':
            annotation = self.annotations.get(filename, None)
            bboxes = []
            labels = []

            if annotation:
                for bbox in annotation['bboxes']:
                    class_id = bbox['class_id']
                    x_min, y_min = bbox['x_min'], bbox['y_min']
                    x_max, y_max = bbox['x_max'], bbox['y_max']
                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            return image, {'bboxes': bboxes, 'labels': labels}

    def _encode_segmentation(self, mask_img):
        mask = np.zeros((mask_img.height, mask_img.width), dtype=np.uint8)
        mask_pixels = np.array(mask_img)

        for rgb, class_id in self.class_mapping.items():
            # Create a mask where this RGB color is present
            match = (
                (mask_pixels[:, :, 0] == rgb[0]) &
                (mask_pixels[:, :, 1] == rgb[1]) &
                (mask_pixels[:, :, 2] == rgb[2])
            )
            mask[match] = class_id

        return torch.tensor(mask, dtype=torch.long)
