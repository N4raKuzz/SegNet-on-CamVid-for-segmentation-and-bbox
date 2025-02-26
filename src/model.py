import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, freeze_backbone=False, mode='segmentation', H=720, W=960, encoder_channels=[512, 256, 128, 64], decoder_channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.mode = mode
        self.H = H
        self.W = W

        # Dynamically create encoders
        self.encoders = nn.ModuleList([ConvBlock(in_channels if i == 0 else encoder_channels[i - 1], c) for i, c in enumerate(encoder_channels)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically create decoders
        self.decoders = nn.ModuleList([DeconvBlock(decoder_channels[i], decoder_channels[i + 1]) for i in range(len(decoder_channels) - 1)])

        self.out_conv_seg = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.out_fc_bbox = nn.Linear(decoder_channels[-1] * H * W, 4)

        self._init_weights()

        if freeze_backbone:
            self._freeze_unet_backbone()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _freeze_unet_backbone(self):
        for module in self.encoders + self.decoders:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_outs.append(x)
            x = self.pool(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            x = x + enc_outs[-(i + 2)]  # Skip connection

        if self.mode == 'segmentation':
            return self.out_conv_seg(x)
        elif self.mode == 'bbox':
            return self.out_fc_bbox(x.view(x.size(0), -1))
