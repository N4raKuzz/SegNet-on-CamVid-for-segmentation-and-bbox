import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetSegDetModel(nn.Module):
    def __init__(self, num_anchors=100, confidence_threshold=0.5, num_seg_classes=2):
        """
        Args:
            num_anchors (int): Number of anchor boxes (i.e. number of detection predictions).
            confidence_threshold (float): Confidence threshold for selecting bounding boxes.
            num_seg_classes (int): Number of segmentation classes.
        """
        super(ResNetSegDetModel, self).__init__()
        self.num_anchors = num_anchors
        self.confidence_threshold = confidence_threshold
        
        # Load pretrained ResNet34 backbone and remove fully connected layers.
        resnet = models.resnet101(pretrained=True)
        # Keep layers until the final convolutional feature map.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # The output of ResNet34 backbone is of shape (B, 512, H/32, W/32).
        
        # Segmentation head:
        self.seg_head = SegNet(num_seg_classes)
        # self.seg_head = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, num_seg_classes, kernel_size=1)
        # )

        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor. Expected shape: either (B, 3, 720, 960)
                              or (B, 720, 960, 3) (in which case it will be permuted).
        Returns:
            seg_logits (torch.Tensor): Segmentation logits of shape (B, num_seg_classes, 720, 960).
            det_preds (torch.Tensor): Detection predictions of shape (B, num_anchors, 5)
                                      where the last dimension is [xmax, xmin, ymax, ymin, confidence].
        """
        # If input is in (B, H, W, C) format, permute it to (B, C, H, W)
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            
        features = self.backbone(x)
        seg_logits = self.seg_head(features)

        # Upsample segmentation logits to the same spatial size as the input image (or mask)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        # print(seg_logits.shape)
        
        # Detection branch:
        # pooled = self.avgpool(features)  # shape: (B, 512, 1, 1)
        # pooled = pooled.view(pooled.size(0), -1)  # shape: (B, 512)
        # det_preds = self.detector(pooled)  # shape: (B, num_anchors*5)
        # det_preds = det_preds.view(-1, self.num_anchors, 5)  # shape: (B, num_anchors, 5)
        
        return seg_logits#, det_preds
    
    def print_head(self, mode="segmentation"):
        """
        Print the model info for classifier head
        """
        if (mode == "segmentation"):
            print(self.seg_head)
        elif (mode == "bbox"):
            print(self.det_head)
        elif (mode == "combined"):
            print(self.det_head)
            print(self.seg_head)

    def postprocess_detections(self, det_preds):
        """
        Post-process detection predictions by applying the confidence threshold.
        
        Args:
            det_preds (torch.Tensor): Detection predictions of shape (B, num_anchors, 5).
                                      The last dimension is [xmax, xmin, ymax, ymin, confidence].
                                      
        Returns:
            output_boxes (list of torch.Tensor): List with length B, each tensor containing the selected
                                                   boxes for that image (each box is a 5-element tensor).
        """
        output_boxes = []
        # Iterate over each sample in the batch.
        for preds in det_preds:
            # Confidence scores are the last element in each prediction.
            conf = preds[:, -1]
            # Select boxes with confidence above threshold.
            keep = conf > self.confidence_threshold
            output_boxes.append(preds[keep])
        return output_boxes

class SegNetEncoder(nn.Module):
    def __init__(self):
        super(SegNetEncoder, self).__init__()
        # Change input channels from 3 to 512
        self.enc1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # # Second encoder block
        # self.enc2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        # x: (B, 3, H, W)
        x1 = self.enc1(x)                     # (B, 64, H, W)
        x1p, indices1 = self.pool1(x1)          # (B, 64, H/2, W/2)
        
        return x1p, (indices1, None), (x1.size(), None)

class SegNetDecoder(nn.Module):
    """
    The Decoder for SegNet.
    Uses max unpooling with stored indices and convolutional blocks to upsample features.
    """
    def __init__(self, num_classes):
        super(SegNetDecoder, self).__init__()
        # First decoder block (corresponding to encoder block 2)
        # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dec2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True)
        # )
        
        # Second decoder block (corresponding to encoder block 1)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 classifier to map to pixel-wise class scores
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x, indices, sizes):
        """
        Args:
            x: Bottleneck feature map from encoder (B, 128, H/4, W/4)
            indices: Tuple (indices1, indices2) from encoder
            sizes: Tuple (size1, size2) of encoder feature maps
        """
        indices1, indices2 = indices
        size1, size2 = sizes
        
        # Decoder block corresponding to encoder block 2
        # x = self.unpool2(x, indices2, output_size=size2)  
        # x = self.dec2(x)                                 
        
        # Decoder block corresponding to encoder block 1
        x = self.unpool1(x, indices1, output_size=size1)  
        x = self.dec1(x)                                  
        
        # Classifier: output logits (B, num_classes, H, W)
        out = self.classifier(x)
        return out

class SegNet(nn.Module):

    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.encoder = SegNetEncoder()
        self.decoder = SegNetDecoder(num_classes)
    
    def forward(self, x):
        # If input is (B, H, W, 3), permute to (B, 3, H, W)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        # Encoder forward
        encoded, indices, sizes = self.encoder(x)
        # Decoder forward
        logits = self.decoder(encoded, indices, sizes)
        return logits
