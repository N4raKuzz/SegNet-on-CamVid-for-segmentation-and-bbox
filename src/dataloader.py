import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Define class names and their corresponding RGB values for segmentation
CLASS_RGB_MAP = {
    "Car": (64, 0, 128),
    "Truck_Bus": (192, 128, 192),
    "MotorcycleScooter": (192, 0, 192),
    "Pedestrian": (64, 64, 0),
    "Bicyclist": (0, 128, 192)
}

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', mode='segmentation', transform=None):
        '''
        Directory structure:    
            data_root/
            camvid/
                train/
                    images/      -> image files (*.png)
                    masks/       -> segmentation mask images (*_L.png)
                val/
                    images/
                    masks/
                test/
                    images/
                    masks/
            annotation/
                COCO JSON file with bounding box annotations
        '''
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.transform = transform

        # Load image paths
        self.image_dir = os.path.join(root_dir, 'camvid', split, 'images')
        self.mask_dir = os.path.join(root_dir, 'camvid', split, 'masks')
        self.annotation_file = os.path.join(root_dir, 'annotation', f'{split}.json')

        self.image_paths = sorted([os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)])

        if self.mode == 'bbox':
            with open(self.annotation_file, 'r') as f:
                self.annotations = {entry['filename']: entry for entry in json.load(f)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'segmentation':
            mask_path = os.path.join(self.mask_dir, filename.replace('.png', '_L.png'))
            mask = Image.open(mask_path).convert('RGB')
            mask = self._encode_segmentation(mask)
            return image, mask

        elif self.mode == 'bbox':
            annotation = self.annotations.get(filename, None)
            bboxes = []
            labels = []

            if annotation:
                for bbox in annotation['bboxes']:
                    if bbox['class'] in CLASS_RGB_MAP:
                        bboxes.append([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']])
                        labels.append(bbox['class'])

            return image, {'bboxes': torch.tensor(bboxes, dtype=torch.float32), 'labels': labels}

    def _encode_segmentation(self, mask_img):
        mask = np.zeros((mask_img.height, mask_img.width), dtype=np.uint8)
        mask_pixels = np.array(mask_img)

        for label, rgb in CLASS_RGB_MAP.items():
            mask[(mask_pixels[:, :, 0] == rgb[0]) &
                 (mask_pixels[:, :, 1] == rgb[1]) &
                 (mask_pixels[:, :, 2] == rgb[2])] = list(CLASS_RGB_MAP.keys()).index(label) + 1

        return torch.tensor(mask, dtype=torch.long)

