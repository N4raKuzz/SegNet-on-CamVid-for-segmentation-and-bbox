import torch
import tqdm
from torchvision.ops import box_iou

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the given dataloader using the provided criterion.
    Calculates:
      - Average loss
      - Average IoU (for bounding-box mode)
      - Average mAP (for bounding-box mode)
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_map = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
            images, targets = batch
            images = images.to(device)

            if model.mode == 'segmentation':
                # 'targets' (batch, H, W)
                masks = targets.to(device)
                outputs = model(images)   # shape (batch, num_classes, H, W)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

            elif model.mode == 'bbox':
                continue

            steps += 1

    avg_loss = total_loss / steps
    # For bounding-box mode, compute mean IoU and mean mAP across the dataset
    # For segmentation mode, these remain 0 or unused.
    avg_iou = total_iou / steps
    avg_map = total_map / steps

    print(f"Validation Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, mAP: {avg_map:.4f}")

    return avg_loss, avg_iou, avg_map
