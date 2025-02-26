import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_model import UNet
from camvid_dataset_loader import CamVidDataset

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
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

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(root_dir, mode='segmentation', num_classes=5, freeze_backbone=False, H=720, W=960, batch_size=4, lr=0.001, encoder_channels=[512, 256, 128, 64], decoder_channels=[64, 128, 256, 512]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = None  # Add any desired transformations here

    test_dataset = CamVidDataset(root_dir=root_dir, split='test', mode=mode, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, num_classes=num_classes, freeze_backbone=freeze_backbone, mode=mode, H=H, W=W, encoder_channels=encoder_channels, decoder_channels=decoder_channels)
    model.to(device)

    weight_filename = f'weights/UNet_mode{mode}_lr{lr}_{len(encoder_channels)}.pth'
    model.load_state_dict(torch.load(weight_filename, map_location=device))
    print(f'Loaded weights from {weight_filename}')

    criterion = nn.CrossEntropyLoss() if mode == 'segmentation' else nn.MSELoss()
    test_loss = evaluate(model, test_loader, criterion, device)

    print(f'Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main(root_dir='data_root', mode='segmentation', num_classes=5, freeze_backbone=False, H=720, W=960, epochs=10, batch_size=4, lr=0.001)
