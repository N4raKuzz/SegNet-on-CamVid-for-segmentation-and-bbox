import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNetSegDetModel
from utils import compute_iou, evaluate_segmentation, evaluate_detection, visualize_predictions
from dataloader import CamVidDataset

# Mapping from evaluation class IDs to names (for printing)
TARGET_CLASS_ID = {
    1: "Bicyclist",
    2: "Car",
    3: "Pedestrian",
    4: "MotorcycleScooter",
    5: "Truck_Bus"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    num_classes = 32       # Train segmentation head over 32 classes.
    seg_threshold = 0.5    # Threshold for generating proposals.
    min_area = 50          # Minimum area for proposals.
    
    # --- Prepare Test Datasets ---
    transform = transforms.ToTensor()
    root_dir = "./data"
    
    # Test dataset for segmentation evaluation
    test_seg_dataset = CamVidDataset(root_dir=root_dir, split='test', mode='segmentation', transform=transform)
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Test dataset for detection evaluation (bbox mode)
    test_det_dataset = CamVidDataset(root_dir=root_dir, split='test', mode='det', transform=transform)
    test_det_loader = DataLoader(test_det_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # --- Iterate over all weight files in the weights/ folder ---
    weight_folder = './weights'
    weight_files = [f for f in os.listdir(weight_folder) if f.endswith('.pth')]
    if len(weight_files) == 0:
        print("No weight files found in", weight_folder)
        return

    for weight_file in weight_files:
        print(f"================ Evaluating Model: {weight_file} =================")
        
        # Initialize model (mode set to "combined" to allow both seg and detection outputs)
        model = ResNetSegDetModel(num_classes=num_classes, mode='combined', seg_threshold=seg_threshold, min_area=min_area)
        model.to(device)
        
        weight_path = os.path.join(weight_folder, weight_file)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Model weights loaded from:", weight_path)
        
        # --- Segmentation Evaluation ---
        # evaluate_segmentation expects the model output tuple; it will take the first element (seg_logits)
        mean_iou, per_class_iou = evaluate_segmentation(model, test_seg_loader, device)
        print(f"Segmentation Evaluation: Mean IoU = {mean_iou:.4f}")
        for class_num, iou_value in zip(TARGET_CLASS_ID.keys(), per_class_iou):
            print(f"  Class {class_num} | {TARGET_CLASS_ID[class_num]}: IoU = {iou_value:.4f}")
        
        # --- Detection Evaluation ---
        # evaluate_detection returns detection mAP along with segmentation IoU computed on detection branch
        ap, det_mean_seg_iou, det_iou_per_class = evaluate_detection(model, test_det_loader, device, iou_threshold=0.5)
        print(f"Detection Evaluation: AP = {ap:.4f}")
        print("Detection segmentation IoU per class:", det_iou_per_class)
        print(f"Detection segmentation Mean IoU = {det_mean_seg_iou:.4f}")
        
        # --- Visualize Predictions ---
        # visualize_predictions can show a few sample predictions from the segmentation branch.
        visualize_predictions(model, test_seg_dataset, device, num_samples=10)
        print("\n")
        
if __name__ == "__main__":
    main()
zy