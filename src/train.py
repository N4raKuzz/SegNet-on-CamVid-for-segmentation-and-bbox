import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import DiceLoss, WeightedIoULoss
from model import ResNetSegDetModel
from dataloader import CamVidDataset
from utils import compute_iou, evaluate_segmentation, evaluate_detection
from torchvision import transforms

def collate_fn_bbox(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

def train(model, optimizer, dice_loss_fn, iou_loss_fn, device, seg_dataloader, bbox_dataloader):
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
        # print(f"masks target {masks.shape}")
        seg_logits = model(images_seg)
        # print(f"seg logits {seg_logits.shape}")
        num_seg_classes = seg_logits.size(1)
        # Convert masks to one-hot encoding: (B, H, W) -> (B, C, H, W)
        masks_one_hot = F.one_hot(masks.long(), num_classes=6).permute(0, 3, 1, 2).float()
        loss_seg = dice_loss_fn(seg_logits, masks_one_hot)

        # # --- Detection branch ---
        # images_bbox = images_bbox.to(device)
        # # Get detection predictions: shape (B, num_anchors, 5)
        # _, det_preds_bbox = model(images_bbox)
        # batch_size = images_bbox.size(0)
        # loss_det_list = []
        # for i in range(batch_size):
        #     pred_boxes = det_preds_bbox[i, :, :4]
        #     pred_boxes_conv = torch.stack([pred_boxes[:, 1], pred_boxes[:, 3],
        #                                    pred_boxes[:, 0], pred_boxes[:, 2]], dim=1)
        #     gt_boxes = targets_bbox[i]['bboxes'].to(device)
        #     _, loss_img = iou_loss_fn(pred_boxes_conv, gt_boxes)
        #     loss_det_list.append(loss_img)
        # # Average the detection loss over the batch
        # loss_det = torch.stack(loss_det_list).mean()

        # total_loss = loss_seg + loss_det
        # total_loss.backward()

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
    num_seg_classes = 6  
    confidence_threshold = 0.6
    lr = 1e-4
    epochs = 20
    batch_size = 5

    # --- Initialize model ---
    model = ResNetSegDetModel(num_anchors=num_anchors,
                                confidence_threshold=confidence_threshold,
                                num_seg_classes=num_seg_classes)
    model.to(device)

    # --- Instantiate loss functions ---
    dice_loss_fn = DiceLoss(smooth=1.0)
    iou_loss_fn = WeightedIoULoss(weight=1.0, eps=1e-6)
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
    # val_bbox_loader = DataLoader(val_bbox_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_bbox)

    # --- Training Loop ---
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        avg_loss_seg, avg_loss_det = train(model, optimizer, dice_loss_fn, iou_loss_fn, device,
                                                     train_seg_loader, train_bbox_loader)
        # print(f"Epoch [{epoch+1}/{epochs}] - Train Dice Loss: {avg_loss_seg:.4f}, Train IoU Loss: {avg_loss_det:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}] - Train Dice Loss: {avg_loss_seg:.4f}")
        
        # --- Validation ---
        # Evaluate segmentation IoU
        mean_iou, per_class_iou = evaluate_segmentation(model, val_seg_loader, device, num_seg_classes)
        # Evaluate detection mAP
        # map_score = evaluate_detection(model, val_bbox_loader, device, iou_threshold=0.5)
        
        epoch_bar.set_postfix({
            'Train Dice': f"{avg_loss_seg:.4f}",
            'Val Mean IoU': f"{mean_iou:.4f}"
        })
        
if __name__ == "__main__":
    main()
