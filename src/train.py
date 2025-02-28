import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from loss import DiceLoss, WeightedIoULoss
from model import ResNet34SegDetModel
from dataloader import CamVidDataset

#########################
# Helper Functions
#########################
def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    """
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6
    return inter_area / union_area

def evaluate_segmentation(model, dataloader, device, num_classes):
    """
    Evaluate segmentation performance by computing mean IoU.
    """
    model.eval()
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)  # shape (B, H, W)
            seg_logits, _ = model(images)
            preds = seg_logits.argmax(dim=1)  # (B, H, W)
            for cls in range(num_classes):
                intersection = ((preds == cls) & (masks == cls)).sum().item()
                union = ((preds == cls) | (masks == cls)).sum().item()
                total_intersections[cls] += intersection
                total_unions[cls] += union
    iou_per_class = total_intersections / (total_unions + 1e-6)
    mean_iou = iou_per_class.mean()
    return mean_iou, iou_per_class

def compute_average_precision(pred_scores, pred_matches, num_gt):
    """
    Compute average precision using the area under the precision-recall curve.
    This is a simplified approximation.
    """
    if len(pred_scores) == 0:
        return 0.0
    # Sort predictions by descending confidence
    sorted_indices = np.argsort(-np.array(pred_scores))
    sorted_matches = np.array(pred_matches)[sorted_indices]
    tp = sorted_matches
    fp = 1 - sorted_matches
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (num_gt + 1e-6)
    # Compute AP as area under the precision-recall curve (using trapezoidal rule)
    ap = np.trapz(precision, recall)
    return ap

def evaluate_detection(model, dataloader, device, iou_threshold=0.5):
    """
    Evaluate detection performance using a simplified mAP calculation.
    (Note: a full mAP implementation is more involved; this is a basic approximation.)
    
    Assumes:
      - The model detection head outputs boxes in format [xmax, xmin, ymax, ymin, confidence].
      - The dataset in bbox mode returns ground truth boxes in [x_min, y_min, x_max, y_max].
    """
    model.eval()
    pred_scores_all = []
    pred_matches_all = []
    total_gt = 0

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            # target is a dict with keys 'bboxes' and 'labels'
            bboxes_gt = target['bboxes']  # list or tensor of shape (B, num_boxes, 4)
            # Forward pass: we only use the detection branch here.
            _, det_preds = model(images)
            batch_size = images.size(0)
            for i in range(batch_size):
                # For each image:
                preds = det_preds[i]  # shape (num_anchors, 5)
                # Filter predictions by confidence threshold.
                confidence_threshold = model.confidence_threshold
                keep = preds[:, -1] > confidence_threshold
                preds = preds[keep]
                if preds.numel() == 0:
                    pred_boxes = np.empty((0, 4))
                    pred_confidences = np.empty((0,))
                else:
                    # Convert boxes from [xmax, xmin, ymax, ymin] to [x_min, y_min, x_max, y_max]
                    pred_boxes = torch.stack([preds[:,1], preds[:,3], preds[:,0], preds[:,2]], dim=1).cpu().numpy()
                    pred_confidences = preds[:, -1].cpu().numpy()
                # Ground truth boxes for image i (assumed already in [x_min, y_min, x_max, y_max])
                gt_boxes = bboxes_gt[i].cpu().numpy() if bboxes_gt[i].numel() > 0 else np.empty((0,4))
                total_gt += len(gt_boxes)
                # For each prediction, determine if it is a true positive.
                image_pred_matches = np.zeros(len(pred_boxes))
                gt_matched = np.zeros(len(gt_boxes))
                for j, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_idx = -1
                    for k, gt_box in enumerate(gt_boxes):
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = k
                    if best_iou >= iou_threshold and best_idx != -1 and gt_matched[best_idx] == 0:
                        image_pred_matches[j] = 1  # true positive
                        gt_matched[best_idx] = 1
                    else:
                        image_pred_matches[j] = 0  # false positive
                pred_scores_all.extend(pred_confidences.tolist())
                pred_matches_all.extend(image_pred_matches.tolist())

    ap = compute_average_precision(pred_scores_all, pred_matches_all, total_gt)
    return ap

#########################
# Training and Validation
#########################
def train_one_epoch(model, optimizer, dice_loss_fn, iou_loss_fn, device,
                    seg_dataloader, bbox_dataloader):
    model.train()
    running_loss_seg = 0.0
    running_loss_det = 0.0
    # Iterate over both dataloaders concurrently.
    # (Assuming both have the same length; otherwise, iterate over the minimum length.)
    for (images_seg, masks), (images_bbox, target_bbox) in zip(seg_dataloader, bbox_dataloader):
        optimizer.zero_grad()
        
        # --- Segmentation branch ---
        images_seg = images_seg.to(device)
        masks = masks.to(device)  # shape (B, H, W)
        seg_logits, _ = model(images_seg)
        num_seg_classes = seg_logits.size(1)
        # Convert masks to one-hot: (B, H, W) -> (B, C, H, W)
        masks_one_hot = F.one_hot(masks.long(), num_classes=num_seg_classes).permute(0, 3, 1, 2).float()
        loss_seg = dice_loss_fn(seg_logits, masks_one_hot)
        
        # --- Detection branch ---
        images_bbox = images_bbox.to(device)
        # target_bbox is a dict with keys 'bboxes' and 'labels'
        # We use only the bounding boxes here.
        bboxes_gt = target_bbox['bboxes'].to(device)  # assumed shape (B, num_boxes, 4)
        # Forward pass for detection.
        _, det_preds = model(images_bbox)
        # det_preds shape: (B, num_anchors, 5) --> take first 4 as boxes.
        pred_boxes = det_preds[..., :4]
        loss_det = iou_loss_fn(pred_boxes, bboxes_gt)
        
        total_loss = loss_seg + loss_det
        total_loss.backward()
        optimizer.step()
        
        running_loss_seg += loss_seg.item()
        running_loss_det += loss_det.item()
        
    avg_loss_seg = running_loss_seg / len(seg_dataloader)
    avg_loss_det = running_loss_det / len(bbox_dataloader)
    return avg_loss_seg, avg_loss_det

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Hyperparameters ---
    num_anchors = 50
    # For CamVid segmentation: background (0) + 5 object classes.
    num_seg_classes = 6  
    confidence_threshold = 0.6
    lr = 1e-3
    epochs = 10
    batch_size = 2

    # --- Initialize model ---
    model = ResNet34SegDetModel(num_anchors=num_anchors,
                                confidence_threshold=confidence_threshold,
                                num_seg_classes=num_seg_classes)
    model.to(device)

    # --- Instantiate loss functions ---
    dice_loss_fn = DiceLoss(smooth=1.0)
    iou_loss_fn = WeightedIoULoss(weight=1.0, eps=1e-6)
    
    # --- Optimizer (only parameters with requires_grad=True, i.e. the heads) ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # --- Create Datasets and Dataloaders ---
    root_dir = "path/to/dataset/root"  # <-- update this path
    # For training, we create two datasets (and dataloaders) using different modes.
    train_seg_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='segmentation', transform=None)
    train_bbox_dataset = CamVidDataset(root_dir=root_dir, split='train', mode='bbox', transform=None)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_bbox_loader = DataLoader(train_bbox_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # For validation, similarly create segmentation and bbox dataloaders.
    val_seg_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='segmentation', transform=None)
    val_bbox_dataset = CamVidDataset(root_dir=root_dir, split='val', mode='bbox', transform=None)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_bbox_loader = DataLoader(val_bbox_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Training Loop ---
    for epoch in range(epochs):
        avg_loss_seg, avg_loss_det = train_one_epoch(model, optimizer, dice_loss_fn, iou_loss_fn, device,
                                                     train_seg_loader, train_bbox_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Dice Loss: {avg_loss_seg:.4f}, Train IoU Loss: {avg_loss_det:.4f}")
        
        # --- Validation ---
        # Evaluate segmentation IoU
        mean_iou, per_class_iou = evaluate_segmentation(model, val_seg_loader, device, num_seg_classes)
        # Evaluate detection mAP
        map_score = evaluate_detection(model, val_bbox_loader, device, iou_threshold=0.5)
        
        print(f"Validation: Mean IoU: {mean_iou:.4f}")
        print(f"Validation: Detection mAP: {map_score:.4f}")
        
if __name__ == "__main__":
    main()
