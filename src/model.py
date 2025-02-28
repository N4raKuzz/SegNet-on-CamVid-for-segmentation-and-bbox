import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet34SegDetModel(nn.Module):
    def __init__(self, num_anchors=100, confidence_threshold=0.5, num_seg_classes=2):
        """
        Args:
            num_anchors (int): Number of anchor boxes (i.e. number of detection predictions).
            confidence_threshold (float): Confidence threshold for selecting bounding boxes.
            num_seg_classes (int): Number of segmentation classes.
        """
        super(ResNet34SegDetModel, self).__init__()
        self.num_anchors = num_anchors
        self.confidence_threshold = confidence_threshold
        
        # Load pretrained ResNet34 backbone and remove fully connected layers.
        resnet = models.resnet34(pretrained=True)
        # Keep layers until the final convolutional feature map.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # The output of ResNet34 backbone is of shape (B, 512, H/32, W/32).
        
        # Segmentation head:
        # Use a few conv layers to refine features and then upsample to the input resolution.
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_seg_classes, kernel_size=1)
        )
        
        # Detection head:
        # Here, we perform global average pooling followed by a linear layer to predict a fixed number of boxes.
        # Each box is represented by 5 numbers: [xmax, xmin, ymax, ymin, confidence]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.detector = nn.Linear(512, num_anchors * 5)
        
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
            
        # Pass through the backbone.
        features = self.backbone(x)  # shape: (B, 512, H/32, W/32)
        
        # Segmentation branch:
        seg_logits = self.seg_head(features)  # shape: (B, num_seg_classes, H/32, W/32)
        # Upsample segmentation logits to the original input size.
        seg_logits = F.interpolate(seg_logits, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Detection branch:
        pooled = self.avgpool(features)  # shape: (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # shape: (B, 512)
        det_preds = self.detector(pooled)  # shape: (B, num_anchors*5)
        det_preds = det_preds.view(-1, self.num_anchors, 5)  # shape: (B, num_anchors, 5)
        
        return seg_logits, det_preds
    
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

# Example usage:
if __name__ == "__main__":
    # Create a dummy input of shape (B, 720, 960, 3)
    dummy_input = torch.randn(2, 720, 960, 3)
    model = ResNet34SegDetModel(num_anchors=50, confidence_threshold=0.6, num_seg_classes=3)
    
    seg_logits, det_preds = model(dummy_input)
    print("Segmentation logits shape:", seg_logits.shape)  # Expected: (2, 3, 720, 960)
    print("Detection predictions shape:", det_preds.shape)  # Expected: (2, 50, 5)
    
    selected_boxes = model.postprocess_detections(det_preds)
    for i, boxes in enumerate(selected_boxes):
        print(f"Image {i} has {boxes.shape[0]} boxes above threshold.")
