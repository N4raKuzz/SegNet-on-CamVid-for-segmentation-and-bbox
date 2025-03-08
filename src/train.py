import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import DiceLoss, IouLoss, FocalLoss, EdgeAwareLoss
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

def train_segmentation(model, optimizer, focal_loss_fn, device, seg_dataloader):
    model.train()
    running_loss_seg = 0.0
    for images, masks in tqdm(seg_dataloader, desc="Segmentation Training"):
        optimizer.zero_grad()
        images = images.to(device)
        masks = masks.to(device)
        # In segmentation-only mode, the model returns seg_logits (det_preds is ignored)
        seg_logits, _ = model(images)
        # One-hot encode masks (assumes 32 segmentation classes)
        masks_one_hot = F.one_hot(masks.long(), num_classes=32).permute(0, 3, 1, 2).float()
        loss_seg = focal_loss_fn(seg_logits, masks_one_hot)
        loss_seg.backward()
        optimizer.step()
        running_loss_seg += loss_seg.item()
    avg_loss_seg = running_loss_seg / len(seg_dataloader)
    return avg_loss_seg

def train_combined(model, optimizer, focal_loss_fn, iou_loss_fn, device, seg_dataloader, det_dataloader, lambda_det):
    model.train()
    running_loss_seg = 0.0
    running_loss_det = 0.0
    # Iterate concurrently over segmentation and detection dataloaders.
    # (Assuming both loaders have roughly the same number of batches.)
    for (images_seg, masks), (images_det, gt_boxes) in tqdm(
            zip(seg_dataloader, det_dataloader), 
            desc="Combined Training",
            total=min(len(seg_dataloader), len(det_dataloader))):
        optimizer.zero_grad()
        images_seg = images_seg.to(device)
        masks = masks.to(device)
        # For shared backbone, we use the segmentation images as input.
        seg_logits, det_preds = model(images_seg)
        print(det_preds)
        # Segmentation loss.
        masks_one_hot = F.one_hot(masks.long(), num_classes=32).permute(0, 3, 1, 2).float()
        loss_seg = focal_loss_fn(seg_logits, masks_one_hot)
        # Detection loss.
        loss_det = 0.0
        count = 0
        # det_preds: list (length B) with each element a tensor of shape (N, 5)
        # gt_boxes: list (length B) of ground-truth boxes for detection.
        for i in range(len(det_preds)):
            if det_preds[i].numel() > 0 and len(gt_boxes[i]) > 0:
                loss_det += iou_loss_fn(det_preds[i][:, :4], gt_boxes[i].to(device))
                count += 1
        if count > 0:
            loss_det = loss_det / count
        else:
            loss_det = 0.0
        total_loss = loss_seg + lambda_det * loss_det
        total_loss.backward()
        optimizer.step()
        running_loss_seg += loss_seg.item()
        running_loss_det += loss_det if isinstance(loss_det, float) else loss_det.item()
    avg_loss_seg = running_loss_seg / min(len(seg_dataloader), len(det_dataloader))
    avg_loss_det = running_loss_det / min(len(seg_dataloader), len(det_dataloader))
    return avg_loss_seg, avg_loss_det

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Hyperparameters ---
    num_seg_classes = 32  # For segmentation training.
    total_epochs = 20     # Total number of epochs.
    seg_only_epochs = 10  # First 30 epochs: segmentation only.
    lr_seg = 1e-4         # Learning rate for segmentation-only phase.
    lr_combined = 1e-5    # Learning rate for combined training phase.
    lambda_det = 1.0      # Balance factor for detection loss during combined training.
    batch_size = 6

    # --- Initialize model ---
    print("Initializing model...")
    # Note: Removed num_anchors and confidence_threshold from the model.
    model = ResNetSegDetModel(num_classes=num_seg_classes, mode='segmentation', seg_threshold=0.5, min_area=50)
    model.to(device)
    model.print_head()

    # --- Instantiate loss functions ---
    # Define alpha for focal loss (for 32 classes). Adjust the five detection class weights accordingly.
    alpha = torch.ones(32)
    alpha[2] = 1.6     # Bicyclist
    alpha[5] = 1.2     # Car
    alpha[16] = 1.6    # Pedestrian
    alpha[13] = 4.0    # MotorcycleScooter
    alpha[27] = 3.0    # Truck_Bus
    alpha_list = alpha.tolist()
    dice_loss_fn = DiceLoss(smooth=1.0)
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=alpha_list)
    iou_loss_fn = IouLoss()
    edge_loss_fn = EdgeAwareLoss()

    # --- Optimizer ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_seg)
    transform = transforms.ToTensor()
    
    # --- Create Datasets and Dataloaders ---
    root_dir = "./data"
    train_seg_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='segmentation', transform=transform)
    train_det_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='det', transform=transform)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_det_loader = DataLoader(train_det_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_bbox)
    
    val_seg_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='segmentation', transform=transform)
    val_det_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='det', transform=transform)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_det_loader = DataLoader(val_det_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_bbox)

    # --- Training Loop ---
    best_val_iou = 0.0
    patience = 10
    counter = 0

    # Create directory for saving weights.
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    for epoch in range(total_epochs):
        print(f"========================== Epoch {epoch+1}/{total_epochs} ===========================")
        if epoch < seg_only_epochs:
            # Segmentation-only training.
            model.select_mode("segmentation")
            # Update optimizer learning rate.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_seg
            avg_loss_seg = train_segmentation(model, optimizer, focal_loss_fn, device, train_seg_loader)
            print(f"[Train]: Epoch {epoch+1} Seg Loss: {avg_loss_seg:.4f}")
        else:
            # Combined training.
            model.select_mode("combined")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_combined
            avg_loss_seg, avg_loss_det = train_combined(
                model, optimizer, focal_loss_fn, iou_loss_fn, device,
                train_seg_loader, train_det_loader, lambda_det
            )
            print(f"[Train]: Epoch {epoch+1} Combined Loss: Seg: {avg_loss_seg:.4f}, Det: {avg_loss_det:.4f}")
        
        # --- Validation (Segmentation Evaluation) ---
        mean_iou, per_class_iou = evaluate_segmentation(model, val_seg_loader, device)
        print(f"[Validation Segmentation]: Epoch {epoch+1} Mean IoU: {mean_iou:.4f}")
        for class_num, iou_value in zip(TARGET_CLASS_ID.keys(), per_class_iou):
            print(f"  Class {class_num} | {TARGET_CLASS_ID[class_num]}: IoU = {iou_value:.4f}")
        
        # --- Validation (Detection Evaluation) ---
        # Use the detection dataloader for evaluation.
        ap, det_mean_seg_iou, det_iou_per_class = evaluate_detection(model, val_det_loader, device, iou_threshold=0.5)
        print(f"[Validation Detection]: Epoch {epoch+1} AP: {ap:.4f}")
        
        # --- Save Best Model and Early Stopping ---
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            counter = 0  # Reset early-stopping counter.
            torch.save(model.state_dict(), './weights/Res34SegDetModel_best.pth')
            print(f"Best model saved with Mean IoU: {best_val_iou:.4f}")
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
