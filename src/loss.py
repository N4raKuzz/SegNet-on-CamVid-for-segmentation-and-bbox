import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    The loss expects the model logits (of shape (B, C, H, W)) and the ground truth 
    in one-hot encoded format (of shape (B, C, H, W)).
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply softmax over the channel dimension.
        pred = F.softmax(pred, dim=1)

        # Flatten the spatial dimensions.
        pred_flat = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.contiguous().view(target.shape[0], target.shape[1], -1)
        
        # Compute the intersection and union.
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        
        # Compute the dice score and loss.
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice  # Dice loss for each class.
        
        # Return the mean loss over classes and batch.
        return loss.mean()

class WeightedIoULoss(nn.Module):
    """
    Weighted IoU Loss for bounding box regression.
    The loss expects predicted and target boxes in the format:
    [xmax, xmin, ymax, ymin] for each box.
    """
    def __init__(self, weight=1.0, eps=1e-6):
        super(WeightedIoULoss, self).__init__()
        if isinstance(weight, list):
            self.weight = torch.tensor(weight, dtype=torch.float32)
        else:
            self.weight = weight
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        # Unpack box coordinates.
        pred_xmax, pred_xmin, pred_ymax, pred_ymin = pred_boxes.unbind(-1)
        target_xmax, target_xmin, target_ymax, target_ymin = target_boxes.unbind(-1)
        
        # Reorder coordinates to standard (xmin, ymin, xmax, ymax).
        inter_xmin = torch.max(pred_xmin, target_xmin)
        inter_ymin = torch.max(pred_ymin, target_ymin)
        inter_xmax = torch.min(pred_xmax, target_xmax)
        inter_ymax = torch.min(pred_ymax, target_ymax)
        
        # Compute intersection area.
        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Compute areas of predicted and target boxes.
        pred_area = (pred_xmax - pred_xmin).clamp(min=0) * (pred_ymax - pred_ymin).clamp(min=0)
        target_area = (target_xmax - target_xmin).clamp(min=0) * (target_ymax - target_ymin).clamp(min=0)
        
        # Compute union area.
        union_area = pred_area + target_area - inter_area + self.eps
        
        # Compute IoU.
        iou = inter_area / union_area
        loss = (1 - iou)
        
        # Apply weight (supports both scalar and tensor weights)
        if isinstance(self.weight, torch.Tensor):
            # Ensure weight is broadcastable to loss.
            loss = self.weight * loss
        else:
            loss = self.weight * loss

        return loss, loss.mean()