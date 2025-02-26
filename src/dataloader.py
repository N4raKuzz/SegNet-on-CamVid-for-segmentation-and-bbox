import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Updated mapping from RGB to integer class IDs
TARGET_CLASS_COLORS = {
    (0, 128, 192): 1,   # Bicyclist
    (64, 0, 128): 2,    # Car
    (64, 64, 0): 3,     # Pedestrian
    (192, 0, 192): 4,   # MotorcycleScooter
    (192, 128, 192): 5  # Truck_Bus
}

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', mode='segmentation', transform=None):
        """
        Directory structure:
            data/
            └── camvid/
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

        :param root_dir: Root directory for the CamVid dataset
        :param split: One of 'train', 'val', or 'test'
        :param mode: 'segmentation' or 'bbox'
        """
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'CamVid', split, 'images')
        self.mask_dir = os.path.join(root_dir, 'CamVid', split, 'masks')
        self.annotation_file = os.path.join(root_dir, 'annotations', f'{split}_annotations.json')
        
        # Load all image paths in a sorted manner
        self.image_paths = sorted([
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith('.png')
        ])
        
        # Load annotations for bbox
        if self.mode == 'bbox':
            with open(self.annotation_file, 'r') as f:
                annotation_list = json.load(f)
                # Convert to dict keyed by filename for easy lookup
                self.annotations = {
                    entry['filename']: entry for entry in annotation_list
                }

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

        elif self.mode == 'bbox':
            annotation = self.annotations.get(filename, None)
            bboxes = []
            labels = []

            if annotation:
                for bbox in annotation['bboxes']:
                    # Use class_id for the label
                    class_id = bbox['class_id']
                    x_min, y_min = bbox['x_min'], bbox['y_min']
                    x_max, y_max = bbox['x_max'], bbox['y_max']

                    # Append the bounding box and the corresponding label
                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            # Convert to tensors
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            return image, {'bboxes': bboxes, 'labels': labels}

    def _encode_segmentation(self, mask_img):
        mask = np.zeros((mask_img.height, mask_img.width), dtype=np.uint8)
        mask_pixels = np.array(mask_img)

        # For each RGB color in TARGET_CLASS_COLORS, assign the corresponding class ID
        for rgb, class_id in TARGET_CLASS_COLORS.items():
            # Create a boolean mask of where this color is found
            match = (
                (mask_pixels[:, :, 0] == rgb[0]) &
                (mask_pixels[:, :, 1] == rgb[1]) &
                (mask_pixels[:, :, 2] == rgb[2])
            )
            mask[match] = class_id

        return torch.tensor(mask, dtype=torch.long)
