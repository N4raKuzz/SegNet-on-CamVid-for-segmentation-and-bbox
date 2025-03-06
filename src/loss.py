import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
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

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, eps=1e-6):
        """
        Args:
            gamma (float): Focusing parameter.
            alpha (float or list or tensor, optional): Weighting factor for classes.
            reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
            eps (float): Small value to avoid numerical instability.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  
        self.eps = eps

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits; target: (B, C, H, W) one-hot
        pred_soft = F.softmax(pred, dim=1)  # convert logits to probabilities
        log_pred = torch.log(pred_soft + self.eps)

        # Compute the probability of the true class at each pixel.
        p_t = (target * pred_soft).sum(dim=1)  # shape: (B, H, W)
        log_p_t = (target * log_pred).sum(dim=1)  # shape: (B, H, W)

        # If alpha (class weighting) is provided, apply it.
        if self.alpha is not None:
            if not isinstance(self.alpha, torch.Tensor):
                alpha = torch.tensor(self.alpha, device=pred.device, dtype=pred.dtype)
            else:
                alpha = self.alpha
            # alpha is expected to be of shape (C,). Multiply per class then sum over channels.
            alpha_factor = (target * alpha.view(1, -1, 1, 1)).sum(dim=1)  # shape: (B, H, W)
        else:
            alpha_factor = 1.0

        # Compute the focal loss.
        loss = -alpha_factor * ((1 - p_t) ** self.gamma) * log_p_t

        return loss.mean()

        
class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
            eps (float): Small constant to avoid division by zero in gradient magnitude computation.
        """
        super(EdgeAwareLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        
        # Define Sobel filters for computing gradients.
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Register the filters as buffers so they move with the module's device.
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits.
            target: (B, C, H, W) one-hot encoded ground truth.
        """
        # Compute probability maps.
        pred_prob = F.softmax(pred, dim=1)
        
        # Compute edge maps for both pred and target.
        pred_edges = self.compute_edge_map(pred_prob)
        target_edges = self.compute_edge_map(target)
        
        # Compute the L1 loss between the edge maps.
        loss = torch.abs(pred_edges - target_edges)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def compute_edge_map(self, tensor):
        # tensor: (B, C, H, W)
        B, C, H, W = tensor.shape
        # Reshape to (B*C, 1, H, W) to apply 2D convolution per channel.
        tensor_reshaped = tensor.view(B * C, 1, H, W)
        
        # Compute gradients using the Sobel filters.
        grad_x = F.conv2d(tensor_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(tensor_reshaped, self.sobel_y, padding=1)
        
        # Compute the gradient magnitude.
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        
        # Reshape back to (B, C, H, W) and average across channels.
        grad_magnitude = grad_magnitude.view(B, C, H, W)
        edge_map = grad_magnitude.mean(dim=1, keepdim=True)  # shape: (B, 1, H, W)
        return edge_map


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