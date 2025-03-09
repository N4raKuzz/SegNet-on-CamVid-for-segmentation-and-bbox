import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import generate_proposals, roi_pooling
class ResNetSegDetModel(nn.Module):
    def __init__(self,
                 num_classes=6,
                 mode='segmentation',
                 seg_threshold=0.4,
                 min_area=50):
        """
        Args:
            num_classes (int): Number of segmentation classes.
            mode (str): 'segmentation', 'det', or 'combined'.
            seg_threshold (float): Threshold for generating proposals from seg logits.
            min_area (int): Minimum connected component area to generate a proposal.
        """
        super(ResNetSegDetModel, self).__init__()
        self.mode = mode
        self.seg_threshold = seg_threshold
        self.min_area = min_area
        
        # Backbone
        resnet = models.resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B, 512, H/32, W/32)
        
        # Segmentation head
        self.seg_head = SegNet(num_classes)  # out_channels = num_classes
        
        # Detection head
        self.det_head = RCNN(in_channels=512, pooled_size=(7,7), scale_factor=1/32)
        self.selected_det_indices = [2, 5, 16, 13, 27]
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (B, 3, H, W).
        Returns:
            seg_logits (torch.Tensor or None)
            det_preds (list[torch.Tensor] or None)
        """
        # Permute if input is (B, H, W, 3)
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            
        # Extract features
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        
        seg_logits = None
        det_preds = None
        
        # Segmentation branch        
        seg_logits = self.seg_head(features)  # (B, num_classes, H/32, W/32)
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Detection branch
        if self.mode != "segmentation":

            selected_logits = seg_logits[:, self.selected_det_indices, :, :]
            det_probs = F.softmax(selected_logits, dim=1)  # (B, 5, H, W)
            proposals_batch = generate_proposals(seg_logits,
                                                    seg_threshold=self.seg_threshold,
                                                    min_area=self.min_area)
            # print(proposals_batch)
            det_preds = self.det_head(features, proposals_batch)
        
        return seg_logits, det_preds
    
    def print_head(self):
        """
        Print the head information of the model
        """
        if self.mode == "segmentation":
            print(self.seg_head)
        elif self.mode == "det":
            print(self.det_head)
        elif self.mode == "combined":
            print("-- Detection Head --")
            print(self.det_head)
            print("-- Segmentation Head --")
            print(self.seg_head)

    def select_mode(self, mode):
        """
        Switch mode from "segmentation" & "det" & "combined"
        """
        self.mode = mode

class SegNetEncoder(nn.Module):
    def __init__(self):
        super(SegNetEncoder, self).__init__()
        self.enc = nn.Sequential(
            # nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.enc(x)                     # (B, 64, H, W)
        xp, indices = self.pool(x)          # (B, 64, H/2, W/2)
        
        return xp, indices, x.size()

class SegNetDecoder(nn.Module):
    """
    The Decoder for SegNet.
    Uses max unpooling with stored indices and convolutional blocks to upsample features.
    """
    def __init__(self, num_classes):
        super(SegNetDecoder, self).__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier to map to pixel-wise class scores
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x, indices, sizes):
        """
        Args:
            x: Bottleneck feature map from encoder (B, 128, H/4, W/4)
            indices: Tuple (indices1, indices2) from encoder
            sizes: Tuple (size1, size2) of encoder feature maps
        """         
        x = self.unpool(x, indices, output_size=sizes)  
        x = self.dec(x)                                  
        
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

class RCNN(nn.Module):
    def __init__(self, in_channels, pooled_size=(7,7), scale_factor=1/32):
        """
        Args:
            in_channels (int): Number of channels in the detection feature map.
            pooled_size (tuple): Output size for ROI pooling.
            scale_factor (float): Factor to scale proposals from input image coordinates to feature map coordinates.
                                  For example, if feature_map = input/32, then scale_factor = 1/32.
        """
        super(RCNN, self).__init__()
        self.pooled_size = pooled_size
        self.scale_factor = scale_factor
        self.fc1 = nn.Linear(in_channels * pooled_size[0] * pooled_size[1], 256)
        self.fc2 = nn.Linear(256, 5)  # 4 offsets and 1 confidence offset
    
    def forward(self, features, proposals):
        """
        Args:
            features (torch.Tensor): Detection feature map of shape (B, C, H_feat, W_feat).
            proposals (list[torch.Tensor]): A list (length B) where each element is a tensor of shape (N, 5)
                                            containing proposals in the original image coordinate space.
        Returns:
            refined_batch (list[torch.Tensor]): A list (length B) where each tensor is of shape (N, 5)
                                                with refined bounding boxes and confidence.
        """
        refined_batch = []
        B = features.size(0)
        for i in range(B):
            props = proposals[i]  # shape: (N, 5)
            if props.numel() == 0:
                refined_batch.append(props)
                continue
            # Scale proposals to feature map coordinates.
            props_scaled = props.clone()
            props_scaled[:, :4] = props_scaled[:, :4] * self.scale_factor
            
            # ROI pooling: extract a fixed-size feature for each proposal.
            rois = roi_pooling(features[i], props_scaled[:, :4], self.pooled_size)
            rois_flat = rois.view(rois.size(0), -1)
            fc1_out = F.relu(self.fc1(rois_flat))
            fc2_out = self.fc2(fc1_out)  # (N, 5): predicted offsets and confidence adjustment
            
            # Refine bounding boxes: add the predicted offsets to the original proposal.
            refined_boxes = props[:, :4] + fc2_out[:, :4]
            # Refine confidence: add the offset to the original confidence and squash with sigmoid.
            refined_conf = torch.sigmoid(props[:, 4:5] + fc2_out[:, 4:5])
            refined = torch.cat([refined_boxes, refined_conf], dim=1)
            refined_batch.append(refined)
        return refined_batch