"""
Minimalist baseline model for extreme precision focus.
Simplest possible architecture to reduce over-segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalistAttention(nn.Module):
    """Ultra-simple attention - just 2 layers."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class SimpleDecoder(nn.Module):
    """Extremely simple decoder to prevent over-segmentation."""
    
    def __init__(self, in_channels=2048):
        super().__init__()
        
        # Single path upsampling - no complex feature mixing
        self.conv_reduce = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        
        # Final prediction layers with heavy regularization
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),  # Heavy dropout
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(16, 1, kernel_size=1),
            # No sigmoid - apply in loss for numerical stability
        )
        
    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.upsample(x)
        x = self.final(x)
        return torch.sigmoid(x)

class MinimalistSegmentationModel(nn.Module):
    """Minimalist model focused on precision."""
    
    def __init__(self, pretrained_model):
        super().__init__()

        # Use only the core feature extractor
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
        
        # Freeze ALL pretrained parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Minimal attention
        self.attention_net = MinimalistAttention()

        # Simple decoder
        self.decoder = SimpleDecoder(in_channels=2048)

    def forward(self, xray, drr):
        # Extract features from X-ray (frozen)
        with torch.no_grad():
            xray_feat = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # Generate simple attention from DRR
        attn_map = self.attention_net(drr)  # [B, 1, 512, 512]
        
        # Resize attention map
        attn_resized = F.interpolate(
            attn_map, 
            size=xray_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 16, 16]
        
        # Very light feature modulation
        modulated_feat = xray_feat * (1 + 0.1 * attn_resized)
        
        # Simple decoding
        seg_out = self.decoder(modulated_feat)
        
        return seg_out, attn_map
