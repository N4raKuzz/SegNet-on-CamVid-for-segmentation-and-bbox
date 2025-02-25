import os, glob, argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#  Dataset Definition
class CamVidDataset(Dataset):
    """
    Assumes the following directory structure:
    
    data_root/
      camvid/
        train/
            images/      -> image files (e.g., *.png)
            masks/       -> segmentation mask images (same filename as images)
        val/
            images/
            masks/
        test/
            images/
            masks/
      annotation/
        COCO JSON file with bounding box annotations
    
    For bounding boxes, each text file is assumed to contain one or more lines.
    Each line should have four numbers (x_min, y_min, x_max, y_max) separated by spaces.
    """
    def __init__(self, root, split='train', transform=None, mask_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_paths = sorted(glob.glob(os.path.join(root, split, "images", "*.png")))
        self.mask_paths  = sorted(glob.glob(os.path.join(root, split, "masks", "*.png")))
        self.bbox_paths  = sorted(glob.glob(os.path.join(root, split, "bboxes", "*.txt")))
        assert len(self.image_paths) == len(self.mask_paths) == len(self.bbox_paths), "Mismatch in data files"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load segmentation mask (assumed single-channel with integer labels)
        mask = Image.open(self.mask_paths[index])
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert to tensor without normalization; squeeze to remove channel if needed
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        # Load bounding box annotations
        with open(self.bbox_paths[index], 'r') as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            coords = [float(x) for x in line.strip().split()]
            boxes.append(coords)  # each is [x_min, y_min, x_max, y_max]
        boxes = torch.tensor(boxes)  # shape: [num_boxes, 4]
        
        return image, mask, boxes