import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNetSegDetModel
from utils import compute_iou, evaluate_segmentation, evaluate_detection, visualize_predictions
from dataloader import CamVidDataset

TARGET_CLASS_ID = {
    1: "Bicyclist",
    2: "Car",
    3: "Pedestrian",
    4: "MotorcycleScooter",
    5: "Truck_Bus"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters (should match training) ---
    num_anchors = 50
    num_seg_classes = 6  # includes background (0) + 5 classes
    confidence_threshold = 0.6
    lr = 5e-5

    # --- Initialize the model ---
    model = ResNetSegDetModel(num_anchors=num_anchors,
                              confidence_threshold=confidence_threshold,
                              num_seg_classes=num_seg_classes)
    model.to(device)

    # --- Load trained weights ---
    weight_path = './weights/Res101SegNet_fullclass_DataAuged_dice.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("Model weights not found at:", weight_path)
        return

    # --- Prepare test dataset ---
    transform = transforms.ToTensor()
    root_dir = "./data"
    # Assumes your CamVidDataset supports a 'test' split.
    test_seg_dataset = CamVidDataset(root_dir=root_dir, split='test', mode='segmentation', transform=transform)
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=1, shuffle=False, num_workers=4)

    mean_iou, per_class_iou = evaluate_segmentation(model, test_seg_loader, device, num_seg_classes)
    print(per_class_iou)
    print(f"[Test Statistic]: \nMean IoU: {mean_iou:.4f}\n")
    for class_num, iou_value in zip(TARGET_CLASS_ID.keys(), per_class_iou):
        print(f"Class {class_num} | {TARGET_CLASS_ID[class_num]}: IoU = {iou_value:.4f}")
    visualize_predictions(model, test_seg_dataset, device, num_samples=10)

if __name__ == "__main__":
    main()
