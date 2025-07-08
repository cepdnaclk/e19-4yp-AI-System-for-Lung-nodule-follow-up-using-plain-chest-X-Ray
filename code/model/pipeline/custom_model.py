import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    """Enhanced attention module with skip connections and batch normalization."""
    
    def __init__(self, in_channels=1, hidden_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # Second conv block
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        # Output layer
        out = torch.sigmoid(self.conv3(out))
        return out

class SegmentationHead(nn.Module):
    """Enhanced segmentation head with progressive upsampling and skip connections."""
    
    def __init__(self, in_channels=2048, hidden_channels=64):
        super().__init__()
        
        # Progressive channel reduction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels * 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels * 4)
        
        self.conv2 = nn.Conv2d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        self.final_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # Progressive upsampling with feature refinement
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Upsample 2x: 16x16 -> 32x32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Upsample 4x: 32x32 -> 128x128
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Final upsampling: 128x128 -> 512x512
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.final_conv(x))
        
        return x

class XrayDRRSegmentationModel(nn.Module):
    """Enhanced X-ray DRR segmentation model with improved architecture."""
    
    def __init__(self, pretrained_model, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        # Feature extractor from pretrained model
        self.feature_extractor = nn.Sequential(
            pretrained_model.model.conv1,
            pretrained_model.model.bn1,
            pretrained_model.model.relu,
            pretrained_model.model.maxpool,
            pretrained_model.model.layer1,
            pretrained_model.model.layer2,
            pretrained_model.model.layer3,
            pretrained_model.model.layer4,
        )
        
        # Freeze pretrained parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Enhanced attention module
        self.attention_net = AttentionModule(in_channels=1, hidden_channels=16)

        # Enhanced segmentation head
        self.segmentation_head = SegmentationHead(in_channels=2048, hidden_channels=64)
        
        # Feature fusion module
        self.fusion_conv = nn.Conv2d(2048, 2048, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(2048)

    def forward(self, xray, drr):
        # Extract features from X-ray
        xray_feat = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # Generate attention map from DRR
        attn_map = self.attention_net(drr)  # [B, 1, 512, 512]
        
        # Resize attention map to match feature map spatial dimensions
        attn_resized = F.interpolate(
            attn_map, 
            size=xray_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 16, 16]
        
        # Feature modulation with attention
        modulated_feat = xray_feat * (1 + self.alpha * attn_resized)
        
        # Additional feature fusion
        fused_feat = F.relu(self.fusion_bn(self.fusion_conv(modulated_feat)))
        
        # Generate segmentation mask
        seg_out = self.segmentation_head(fused_feat)  # [B, 1, 512, 512]
        
        return seg_out
