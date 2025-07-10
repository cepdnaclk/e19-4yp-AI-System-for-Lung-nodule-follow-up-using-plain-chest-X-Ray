import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    """Enhanced spatial attention module using DRR prior knowledge."""
    
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Feature fusion
        self.fusion = nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(hidden_channels)
        
        # Attention refinement
        self.refine1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, drr):
        # Multi-scale feature extraction
        f1 = F.relu(self.bn1(self.conv1(drr)))
        f2 = F.relu(self.bn2(self.conv2(f1)))
        f3 = F.relu(self.bn3(self.conv3(f2)))
        
        # Concatenate multi-scale features
        fused = torch.cat([f1, f2, f3], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion(fused)))
        fused = self.dropout(fused)
        
        # Attention refinement
        attn = F.relu(self.refine1(fused))
        attn = self.dropout(attn)
        attn = torch.sigmoid(self.refine2(attn))
        
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
    """Enhanced segmentation head with progressive upsampling and feature pyramid."""
    
    def __init__(self, in_channels=2048, hidden_channels=128):
        super().__init__()
        
        # Feature Pyramid Network (FPN) style architecture
        self.lateral1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.lateral3 = nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=1)
        
        # Refinement layers
        self.refine1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1)
        self.refine3 = nn.Conv2d(hidden_channels // 4, hidden_channels // 4, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.bn3 = nn.BatchNorm2d(hidden_channels // 4)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(hidden_channels // 4, 1, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        # Progressive upsampling with refinement
        # Stage 1: 16x16 -> 32x32
        x1 = self.lateral1(x)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = F.relu(self.bn1(self.refine1(x1)))
        x1 = self.dropout(x1)
        
        # Stage 2: 32x32 -> 64x64  
        x2 = self.lateral2(x1)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = F.relu(self.bn2(self.refine2(x2)))
        x2 = self.dropout(x2)
        
        # Stage 3: 64x64 -> 128x128
        x3 = self.lateral3(x2)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = F.relu(self.bn3(self.refine3(x3)))
        x3 = self.dropout(x3)
        
        # Final upsampling: 128x128 -> 512x512
        x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
        output = torch.sigmoid(self.final_conv(x3))
        
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
        
        # Freeze pretrained parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Enhanced attention module using DRR
        self.spatial_attention = SpatialAttentionModule(in_channels=1, hidden_channels=32)
        
        # Channel attention for feature enhancement
        if use_channel_attention:
            self.channel_attention = ChannelAttentionModule(in_channels=2048)

        # Enhanced segmentation head
        self.segmentation_head = EnhancedSegmentationHead(in_channels=2048, hidden_channels=128)
        
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
        seg_out = self.segmentation_head(fused_feat)  # [B, 1, 512, 512]
        
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
