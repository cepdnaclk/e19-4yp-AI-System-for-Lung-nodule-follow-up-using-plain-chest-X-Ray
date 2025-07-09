"""
Simplified and improved DRR segmentation model with focus on accuracy.
Removes unnecessary complexity while maintaining effective architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttentionModule(nn.Module):
    """Simplified attention module focused on effectiveness."""
    
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        # Deeper network with more capacity
        self.net = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections for better spatial preservation."""
    
    def __init__(self, in_channels=2048):
        super().__init__()
        
        # Progressive upsampling with skip connections
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.up5 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Progressive upsampling: 16x16 -> 512x512
        x = self.up1(x)    # 16x16 -> 32x32
        x = self.conv1(x)
        
        x = self.up2(x)    # 32x32 -> 64x64
        x = self.conv2(x)
        
        x = self.up3(x)    # 64x64 -> 128x128
        x = self.conv3(x)
        
        x = self.up4(x)    # 128x128 -> 256x256
        x = self.conv4(x)
        
        x = self.up5(x)    # 256x256 -> 512x512
        x = self.final(x)
        
        return x

class ImprovedXrayDRRSegmentationModel(nn.Module):
    """Improved X-ray DRR segmentation model with simplified architecture."""
    
    def __init__(self, pretrained_model, alpha=0.3):
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
        
        # Partially freeze pretrained parameters (unfreeze last layer for adaptation)
        for param in self.feature_extractor[:-1].parameters():
            param.requires_grad = False
        
        # Allow last layer to adapt
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        # Improved attention module
        self.attention_net = SimpleAttentionModule(in_channels=1, hidden_channels=32)

        # U-Net style decoder
        self.decoder = UNetDecoder(in_channels=2048)

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
        
        # Generate segmentation mask using U-Net decoder
        seg_out = self.decoder(modulated_feat)  # [B, 1, 512, 512]
        
        return seg_out, attn_map
