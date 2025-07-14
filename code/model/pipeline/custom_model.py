import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    """Enhanced spatial attention module using DRR prior knowledge with deeper architecture."""
    
    def __init__(self, in_channels=1, hidden_channels=64):  # Increased from 32 to 64
        super().__init__()
        # Multi-scale feature extraction with deeper processing
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Enhanced feature fusion with residual connections
        self.fusion = nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(hidden_channels)
        
        # Deeper attention refinement network
        self.refine1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.refine3 = nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1)
        self.refine4 = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)
        
        # Additional batch normalization for deeper network
        self.refine_bn1 = nn.BatchNorm2d(hidden_channels)
        self.refine_bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.refine_bn3 = nn.BatchNorm2d(hidden_channels // 4)
        
        self.dropout = nn.Dropout2d(0.15)  # Slightly increased dropout
        
    def forward(self, drr):
        # Multi-scale feature extraction
        f1 = F.relu(self.bn1(self.conv1(drr)))
        f2 = F.relu(self.bn2(self.conv2(f1)))
        f3 = F.relu(self.bn3(self.conv3(f2)))
        
        # Concatenate multi-scale features
        fused = torch.cat([f1, f2, f3], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion(fused)))
        fused = self.dropout(fused)
        
        # Deeper attention refinement with residual connections
        attn = F.relu(self.refine_bn1(self.refine1(fused))) + fused  # Residual connection
        attn = self.dropout(attn)
        attn = F.relu(self.refine_bn2(self.refine2(attn)))
        attn = self.dropout(attn)
        attn = F.relu(self.refine_bn3(self.refine3(attn)))
        attn = torch.sigmoid(self.refine4(attn))
        
        return attn

class ChannelAttentionModule(nn.Module):
    """Channel attention to enhance important feature channels."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out) * x

class EnhancedSegmentationHead(nn.Module):
    """Enhanced segmentation head with transposed convolutions and skip connections."""
    
    def __init__(self, in_channels=2048, hidden_channels=256):  # Increased hidden channels
        super().__init__()
        
        # Transposed convolutions for learnable upsampling (instead of bilinear interpolation)
        self.upconv1 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(hidden_channels // 4, hidden_channels // 8, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(hidden_channels // 8, hidden_channels // 16, kernel_size=2, stride=2)
        
        # Refinement layers after each upsampling
        self.refine1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.refine2 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.refine3 = nn.Sequential(
            nn.Conv2d(hidden_channels // 4, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.refine4 = nn.Sequential(
            nn.Conv2d(hidden_channels // 8, hidden_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 8, hidden_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 8),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels // 16, hidden_channels // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels // 32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        # Learnable upsampling with transposed convolutions
        # Stage 1: 16x16 -> 32x32
        x1 = self.upconv1(x)  # [B, 256, 32, 32]
        x1 = self.refine1(x1)
        x1 = self.dropout(x1)
        
        # Stage 2: 32x32 -> 64x64  
        x2 = self.upconv2(x1)  # [B, 128, 64, 64]
        x2 = self.refine2(x2)
        x2 = self.dropout(x2)
        
        # Stage 3: 64x64 -> 128x128
        x3 = self.upconv3(x2)  # [B, 64, 128, 128]
        x3 = self.refine3(x3)
        x3 = self.dropout(x3)
        
        # Stage 4: 128x128 -> 256x256
        x4 = self.upconv4(x3)  # [B, 32, 256, 256]
        x4 = self.refine4(x4)
        x4 = self.dropout(x4)
        
        # Stage 5: 256x256 -> 512x512
        x5 = self.upconv5(x4)  # [B, 16, 512, 512]
        
        # Final prediction
        output = self.final_conv(x5)  # [B, 1, 512, 512]
        
        return output

class ImprovedXrayDRRSegmentationModel(nn.Module):
    """Improved X-ray DRR segmentation model with better attention and feature fusion."""
    
    def __init__(self, pretrained_model, alpha=0.3, use_channel_attention=True):
        super().__init__()
        self.alpha = alpha
        self.use_channel_attention = use_channel_attention

        # Feature extractor from pretrained model (frozen)
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
        
        # Partially freeze pretrained parameters (allow more adaptation)
        # Freeze early layers but allow last 2 layers to adapt to medical domain
        for param in self.feature_extractor[:-2].parameters():
            param.requires_grad = False
        
        # Last two layers (layer3 and layer4) are trainable for domain adaptation
        for param in self.feature_extractor[-2:].parameters():
            param.requires_grad = True

        # Enhanced attention module using DRR with increased capacity
        self.spatial_attention = SpatialAttentionModule(in_channels=1, hidden_channels=64)
        
        # Channel attention for feature enhancement
        if use_channel_attention:
            self.channel_attention = ChannelAttentionModule(in_channels=2048)

        # Enhanced segmentation head with transposed convolutions
        self.segmentation_head = EnhancedSegmentationHead(in_channels=2048, hidden_channels=256)
        
        # Feature fusion modules
        self.fusion_conv1 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv2d(2048, 2048, kernel_size=1)
        self.fusion_bn1 = nn.BatchNorm2d(2048)
        self.fusion_bn2 = nn.BatchNorm2d(2048)
        
        # Gating mechanism for attention
        self.gate = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, xray, drr):
        # Extract features from X-ray using pretrained model
        xray_feat = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # Generate spatial attention map from DRR
        spatial_attn = self.spatial_attention(drr)  # [B, 1, 512, 512]
        
        # Resize attention map to match feature map spatial dimensions
        spatial_attn_resized = F.interpolate(
            spatial_attn, 
            size=xray_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 16, 16]
        
        # Apply channel attention if enabled
        if self.use_channel_attention:
            xray_feat = self.channel_attention(xray_feat)
        
        # Feature modulation with gated attention
        gate_weights = self.gate(xray_feat)
        modulated_feat = xray_feat * (1 + self.alpha * spatial_attn_resized * gate_weights)
        
        # Feature fusion with residual connection
        fused_feat = F.relu(self.fusion_bn1(self.fusion_conv1(modulated_feat)))
        fused_feat = F.relu(self.fusion_bn2(self.fusion_conv2(fused_feat))) + modulated_feat
        
        # Generate segmentation mask
        seg_out = self.segmentation_head(modulated_feat)  # [B, 1, 512, 512]
        
        # Return both segmentation output and attention map for visualization
        return {
            'segmentation': seg_out,
            'attention': spatial_attn,
            'features': fused_feat
        }

# Backward compatibility alias
class XrayDRRSegmentationModel(ImprovedXrayDRRSegmentationModel):
    """Backward compatibility wrapper."""
    
    def __init__(self, pretrained_model, alpha=0.3):
        super().__init__(pretrained_model, alpha)
        # Keep old attention module name for compatibility
        self.attention_net = self.spatial_attention
        
    def forward(self, xray, drr):
        result = super().forward(xray, drr)
        # Return only segmentation for backward compatibility
        return result['segmentation']

class MultiScaleAttentionModule(nn.Module):
    """Multi-scale spatial attention module for better nodule detection at different scales."""
    
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()
        # Different scale processing branches
        self.scale_4 = self._make_scale_branch(in_channels, hidden_channels, scale=4)
        self.scale_2 = self._make_scale_branch(in_channels, hidden_channels, scale=2)
        self.scale_1 = self._make_scale_branch(in_channels, hidden_channels, scale=1)
        
        # Feature fusion across scales
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _make_scale_branch(self, in_channels, hidden_channels, scale):
        """Create a processing branch for a specific scale."""
        layers = []
        if scale > 1:
            layers.append(nn.AvgPool2d(kernel_size=scale, stride=scale))
        
        layers.extend([
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Process at different scales
        feat_4 = self.scale_4(x)
        feat_4 = F.interpolate(feat_4, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        feat_2 = self.scale_2(x)
        feat_2 = F.interpolate(feat_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        feat_1 = self.scale_1(x)
        
        # Fuse multi-scale features
        fused = torch.cat([feat_4, feat_2, feat_1], dim=1)
        attention = self.fusion(fused)
        
        return attention
