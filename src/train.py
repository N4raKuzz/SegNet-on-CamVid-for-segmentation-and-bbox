import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNet
from dataloader import CamVidDataset

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        optimizer.zero_grad()

        if model.mode == 'segmentation':
            masks = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
        else:  # bbox
            bboxes = targets['bboxes'].to(device)
            outputs = model(images)
            loss = criterion(outputs, bboxes)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Main script
def main(root_dir, mode='segmentation', num_classes=5, freeze_backbone=False, H=256, W=256, epochs=10, batch_size=4, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = None  # Add any desired transformations here

    train_dataset = CamVidDataset(root_dir=root_dir, split='train', mode=mode, transform=transform)
    val_dataset = CamVidDataset(root_dir=root_dir, split='val', mode=mode, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=3, num_classes=num_classes, freeze_backbone=freeze_backbone, mode=mode, H=H, W=W)
    model.to(device)

    criterion = nn.CrossEntropyLoss() if mode == 'segmentation' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

if __name__ == '__main__':
    main(root_dir='data_root', mode='segmentation', num_classes=5, freeze_backbone=False, H=720, W=960, epochs=10, batch_size=4, lr=0.001)
