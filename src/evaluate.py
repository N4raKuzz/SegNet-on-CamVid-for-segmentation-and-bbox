import torch
import tqdm
from torchvision.ops import box_iou

# IoU Calculation
def calculate_iou(pred_boxes, true_boxes):
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0
    ious = box_iou(pred_boxes, true_boxes)
    return ious.mean().item()

# mAP Calculation
def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0
    ious = box_iou(pred_boxes, true_boxes)
    tp = (ious > iou_threshold).sum().item()
    return tp / len(true_boxes)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_map = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)

            if model.mode == 'segmentation':
                masks = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
            else:  # bbox
                bboxes = targets['bboxes'].to(device)
                outputs = model(images)
                loss = criterion(outputs, bboxes)
                total_iou += calculate_iou(outputs, bboxes)
                total_map += calculate_map(outputs, bboxes)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_map = total_map / len(dataloader)
    return avg_loss, avg_iou, avg_map