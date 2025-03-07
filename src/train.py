import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import DiceLoss, WeightedIoULoss, FocalLoss
from model import ResNetSegDetModel
from dataloader import CamVidDataset
from utils import compute_iou, evaluate_segmentation, evaluate_detection
from torchvision import transforms

TARGET_CLASS_ID = {
    1: "Bicyclist",
    2: "Car",
    3: "Pedestrian",
    4: "MotorcycleScooter",
    5: "Truck_Bus"
}

def collate_fn_bbox(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

def train(model, optimizer, focal_loss_fn, iou_loss_fn, device, seg_dataloader, bbox_dataloader):
    model.train()
    running_loss_seg = 0.0
    running_loss_det = 0.0

    # Iterate concurrently over segmentation and bbox dataloaders.
    for images_seg, masks in seg_dataloader:
    # for (images_seg, masks), (images_bbox, targets_bbox) in zip(seg_dataloader, bbox_dataloader):
        optimizer.zero_grad()

        # --- Segmentation branch ---
        images_seg = images_seg.to(device)
        masks = masks.to(device)  # shape: (B, H, W)
        seg_logits = model(images_seg)
        num_seg_classes = seg_logits.size(1)
        masks_one_hot = F.one_hot(masks.long(), num_classes=32).permute(0, 3, 1, 2).float() # To one-hot (B, H, W) -> (B, C, H, W)
        loss_seg = focal_loss_fn(seg_logits, masks_one_hot)

        # --- Detection branch ---

        total_loss = loss_seg + loss_det
        total_loss.backward()

        loss_seg.backward()
        optimizer.step()

        running_loss_seg += loss_seg.item()
        # running_loss_det += loss_det.item()

    avg_loss_seg = running_loss_seg / len(seg_dataloader)
    avg_loss_det = running_loss_det / len(bbox_dataloader)

    # return avg_loss_seg, avg_loss_det
    return avg_loss_seg, None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Hyperparameters ---
    num_anchors = 50
    num_seg_classes = 32 
    confidence_threshold = 0.6
    lr = 1e-4
    epochs = 30
    batch_size = 6

    # --- Initialize model ---
    print("Initializing model...")
    model = ResNetSegDetModel(num_anchors=num_anchors,
                                confidence_threshold=confidence_threshold,
                                num_seg_classes=num_seg_classes)
    model.to(device)
    model.print_head()

    # --- Instantiate loss functions ---
    alpha = torch.ones(32)
    alpha[2] = 1.6     # Bicyclist
    alpha[5] = 1.2     # Car
    alpha[16] = 1.6    # Pedestrian
    alpha[13] = 4.0    # MotorcycleScooter
    alpha[27] = 3.0    # Truck_Bus
    alpha_list = alpha.tolist()
    dice_loss_fn = DiceLoss(smooth=1.0)
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=alpha_list)
    iou_loss_fn = WeightedIoULoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    transform = transforms.ToTensor()
    
    # --- Create Datasets and Dataloaders ---
    root_dir = "./data"
    
    train_seg_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='segmentation', transform=transform)
    train_bbox_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='bbox', transform=transform)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_bbox_loader = DataLoader(train_bbox_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_bbox)
    
    val_seg_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='segmentation', transform=transform)
    # val_bbox_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='bbox', transform=transform)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # val_bbox_loader = DataLoader(val_wbbox_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_bbox)

    # --- Training Loop ---
    best_val_iou = 0.0
    patience = 10
    counter = 0

    # Ensure the weights directory exists
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    for epoch in range(epochs):
        print(f"========================== Epoch {epoch+1} ===========================")
        
        # --- Training ---
        avg_loss_seg, avg_loss_det = train(model, optimizer, dice_loss_fn, focal_loss_fn, iou_loss_fn, device,
                                        train_seg_loader, train_bbox_loader)
        print(f"[Train Statistic]: \nEpoch [{epoch+1}/{epochs}] - Train Combined Loss: {avg_loss_seg:.4f}\n")
        
        # --- Validation ---
        mean_iou, per_class_iou = evaluate_segmentation(model, val_seg_loader, device)        
        print(f"[Validation Statistic]: \nEpoch [{epoch+1}/{epochs}] - Validation Mean IoU: {mean_iou:.4f}\n")
        for class_num, iou_value in zip(TARGET_CLASS_ID.keys(), per_class_iou):
            print(f"Class {class_num} | {TARGET_CLASS_ID[class_num]}: IoU = {iou_value:.4f}")
        
        # --- Save Best Model and Early Stopping ---
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            counter = 0  # reset counter on improvement
            # Save best model state
            torch.save(model.state_dict(), f'./weights/Res34SegNet_fullclass_DataAuged_combined3.pth')
            print(f"Best model saved with Validation Mean IoU: {best_val_iou:.4f}")
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

        
if __name__ == "__main__":
    main()
