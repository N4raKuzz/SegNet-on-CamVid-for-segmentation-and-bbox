import torch
import torch.nn as nn

class WeightedIoULoss(nn.Module):
    def __init__(self, class_weights=None, smooth=1e-6, num_class=5):
        """
        :param class_weights: list or tensor of weights for each class; length should equal num_classes.
                              If None, defaults to equal weighting.
        :param smooth: smoothing constant to avoid division by zero.
        """
        super(WeightedIoULoss, self).__init__()
        if class_weights is None:
            self.class_weights = [1.0] * num_class 
        else:
            self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        Computes the weighted IoU loss between model outputs and targets.
        :param outputs: raw logits from the model with shape (batch, num_classes, H, W)
        :param targets: ground truth segmentation masks with shape (batch, H, W) containing class indices
        :return: weighted IoU loss
        """
        num_classes = outputs.size(1)
        # Convert logits to probabilities
        probs = torch.softmax(outputs, dim=1)
        # Convert targets to one-hot encoding: (batch, H, W, num_classes) -> (batch, num_classes, H, W)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(num_classes):
            pred_flat = probs[:, c, :, :].contiguous().view(-1)
            target_flat = targets_one_hot[:, c, :, :].contiguous().view(-1)
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            loss += self.class_weights[c] * (1 - iou)

        return loss / num_classes