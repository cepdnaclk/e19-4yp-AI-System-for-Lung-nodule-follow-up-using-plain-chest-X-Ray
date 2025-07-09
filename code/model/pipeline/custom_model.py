import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    """Simplified but more effective attention module."""
    
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        # Deeper network with more capacity for better attention
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels // 2)
        
        self.output = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.output(x))
        return x

class SegmentationHead(nn.Module):
    """Improved segmentation head with better upsampling."""
    
    def __init__(self, in_channels=2048, hidden_channels=128):
        super().__init__()
        
        # First upsampling block: 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(in_channels, hidden_channels*4, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels*4, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # Second upsampling block: 32x32 -> 64x64  
        self.up2 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels*2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Third upsampling block: 64x64 -> 128x128
        self.up3 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # Fourth upsampling block: 128x128 -> 256x256
        self.up4 = nn.ConvTranspose2d(hidden_channels//2, hidden_channels//2, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_channels//2, hidden_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling: 256x256 -> 512x512
        self.up5 = nn.ConvTranspose2d(hidden_channels//4, hidden_channels//4, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels//4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self.conv4(x)
        
        x = self.up5(x)
        x = self.final(x)
        
        return x

class XrayDRRSegmentationModel(nn.Module):
    """Improved X-ray DRR segmentation model with cleaner architecture."""
    
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
        
        # Partially freeze pretrained parameters (allow some adaptation)
        for param in self.feature_extractor[:-1].parameters():
            param.requires_grad = False
        
        # Allow last layer to adapt to medical domain
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        # Improved attention module  
        self.attention_net = AttentionModule(in_channels=1, hidden_channels=32)

        # Better segmentation head
        self.segmentation_head = SegmentationHead(in_channels=2048, hidden_channels=128)

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
        
        # Feature modulation with attention (reduced alpha for stability)
        modulated_feat = xray_feat * (1 + self.alpha * attn_resized)
        
        # Generate segmentation mask
        seg_out = self.segmentation_head(modulated_feat)  # [B, 1, 512, 512]
        
        return seg_out, attn_map  # Return both outputs for loss calculation
