"""
Improved attention mechanism with better anatomical awareness and alignment.
This addresses the issue where attention focuses on the wrong areas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicallyAwareAttentionModule(nn.Module):
    """Attention module that uses both DRR and X-ray features for better alignment."""
    
    def __init__(self, drr_channels=1, xray_feature_channels=2048, hidden_channels=32):
        super().__init__()
        
        # DRR-based spatial attention pathway
        self.drr_attention = nn.Sequential(
            nn.Conv2d(drr_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_channels//2, 1, kernel_size=3, padding=1)
        )
        
        # X-ray feature-based attention pathway (helps with anatomical alignment)
        self.xray_attention = nn.Sequential(
            # Upsample X-ray features to match DRR size
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),  # 16x16 -> 512x512
            nn.Conv2d(xray_feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(hidden_channels//2, 1, kernel_size=1)
        )
        
        # Fusion network to combine both attention sources
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, hidden_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//4, 1, kernel_size=1)
        )
        
        # Attention refinement network
        self.attention_refine = nn.Sequential(
            nn.Conv2d(1, hidden_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//4, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, drr, xray_features):
        """
        Args:
            drr: DRR image [B, 1, 512, 512]
            xray_features: X-ray features from backbone [B, 2048, 16, 16]
        """
        # Generate attention from DRR
        drr_att = self.drr_attention(drr)  # [B, 1, 512, 512]
        
        # Generate attention from X-ray features
        xray_att = self.xray_attention(xray_features)  # [B, 1, 512, 512]
        
        # Combine both attention sources
        combined_att = torch.cat([drr_att, xray_att], dim=1)  # [B, 2, 512, 512]
        fused_att = self.attention_fusion(combined_att)  # [B, 1, 512, 512]
        
        # Refine the attention
        refined_att = self.attention_refine(fused_att)  # [B, 1, 512, 512]
        final_att = fused_att + refined_att  # Residual connection
        
        # Apply improved normalization
        final_att = self._normalize_attention(final_att)
        
        return final_att
    
    def _normalize_attention(self, attention):
        """Improved attention normalization for better spatial contrast."""
        B, C, H, W = attention.shape
        
        # Apply spatial softmax but with temperature scaling for sharpness
        attention_flat = attention.view(B, C, -1)  # [B, 1, H*W]
        
        # Use temperature to control attention sharpness
        temperature = 0.5  # Lower = sharper attention
        attention_scaled = attention_flat / temperature
        
        # Apply softmax
        attention_normalized = F.softmax(attention_scaled, dim=2)
        
        # Reshape back
        attention_map = attention_normalized.view(B, C, H, W)
        
        # Scale to make visible and add some contrast
        attention_map = attention_map * (H * W * 0.1)  # Scale factor for visibility
        attention_map = torch.clamp(attention_map, 0, 1)
        
        # Add some sharpening
        attention_map = torch.pow(attention_map, 0.8)  # Gamma correction for contrast
        
        return attention_map

class SupervisedAttentionModule(nn.Module):
    """Attention module that can be trained with supervision from ground truth masks."""
    
    def __init__(self, in_channels=1, hidden_channels=32):
        super().__init__()
        
        # Attention network with residual connections
        self.attention_net = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_channels),
            self._make_residual_block(hidden_channels),
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, 1, kernel_size=1)
        )
        
    def _make_residual_block(self, channels):
        """Create a residual block for attention refinement."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
    def forward(self, drr):
        """Forward pass through supervised attention."""
        x = self.attention_net(drr)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = F.relu(x)
        
        # Generate attention map
        attention = self.output_layer(x)
        
        # Apply sigmoid for [0,1] range
        attention = torch.sigmoid(attention)
        
        return attention
    
    def attention_loss(self, predicted_attention, ground_truth_mask):
        """Compute supervised attention loss."""
        # Resize ground truth to match attention if needed
        if predicted_attention.shape != ground_truth_mask.shape:
            gt_resized = F.interpolate(
                ground_truth_mask,
                size=predicted_attention.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            gt_resized = ground_truth_mask
        
        # Binary cross-entropy loss for attention supervision
        attention_bce = F.binary_cross_entropy(predicted_attention, gt_resized)
        
        # Add focal loss component for hard examples
        alpha = 0.25
        gamma = 2.0
        pt = torch.where(gt_resized == 1, predicted_attention, 1 - predicted_attention)
        focal_weight = alpha * torch.pow(1 - pt, gamma)
        focal_loss = focal_weight * F.binary_cross_entropy(predicted_attention, gt_resized, reduction='none')
        focal_loss = focal_loss.mean()
        
        # Combine losses
        total_attention_loss = attention_bce + 0.5 * focal_loss
        
        return total_attention_loss

class ImprovedXrayDRRModel(nn.Module):
    """Improved model with better attention mechanism and training strategy."""
    
    def __init__(self, pretrained_model=None, alpha=0.3, use_supervised_attention=True, target_pathology='Nodule'):
        super().__init__()
        self.alpha = alpha
        self.use_supervised_attention = use_supervised_attention
        self.target_pathology = target_pathology
        
        # Initialize pretrained model
        if pretrained_model is None:
            import torchxrayvision as xrv
            pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        
        self.pretrained_model = pretrained_model
        
        # Get pathology index
        self.pathology_idx = None
        if hasattr(pretrained_model, 'pathologies'):
            try:
                self.pathology_idx = pretrained_model.pathologies.index(target_pathology)
                print(f"Found {target_pathology} at index {self.pathology_idx}")
            except ValueError:
                similar_pathologies = [p for p in pretrained_model.pathologies 
                                     if target_pathology.lower() in p.lower()]
                if similar_pathologies:
                    self.pathology_idx = pretrained_model.pathologies.index(similar_pathologies[0])
                    print(f"Using similar pathology: {similar_pathologies[0]} at index {self.pathology_idx}")
        
        # Feature extractor
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
        
        # Freeze early layers
        for param in self.feature_extractor[:-1].parameters():
            param.requires_grad = False
        
        # Nodule-specific feature adaptation
        self.nodule_feature_extractor = self._create_nodule_feature_extractor()
        
        # Choose attention mechanism
        if use_supervised_attention:
            self.spatial_attention = SupervisedAttentionModule(in_channels=1, hidden_channels=32)
        else:
            self.spatial_attention = AnatomicallyAwareAttentionModule(
                drr_channels=1, 
                xray_feature_channels=2048,
                hidden_channels=32
            )
        
        # Feature fusion and adaptation
        self.nodule_adaptation = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Improved segmentation head with skip connections
        self.segmentation_head = self._create_segmentation_head()
        
    def _create_nodule_feature_extractor(self):
        """Create nodule-specific feature extractor."""
        if hasattr(self.pretrained_model, 'classifier') and self.pathology_idx is not None:
            classifier_weights = self.pretrained_model.classifier.weight.data
            nodule_weights = classifier_weights[self.pathology_idx:self.pathology_idx+1]
            
            nodule_conv = nn.Conv2d(2048, 1, kernel_size=1, bias=False)
            nodule_conv.weight.data = nodule_weights.unsqueeze(-1).unsqueeze(-1)
            nodule_conv.weight.requires_grad = False
            
            return nn.Sequential(
                nodule_conv,
                nn.Sigmoid(),
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(2048, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3)
            )
    
    def _create_segmentation_head(self):
        """Create improved segmentation head with skip connections."""
        return nn.Sequential(
            # Initial processing
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Upsampling with better interpolation
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16->32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32->64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),  # 64->512
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(8, 1, kernel_size=1)
        )
    
    def forward(self, xray, drr, ground_truth_mask=None):
        """Forward pass with optional supervised attention training."""
        # Extract X-ray features
        xray_feat = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # Extract nodule-specific features
        nodule_specific_feat = self.nodule_feature_extractor(xray_feat)  # [B, 64, 16, 16]
        
        # Adapt features
        adapted_feat = self.nodule_adaptation(xray_feat)  # [B, 128, 16, 16]
        
        # Generate spatial attention
        if self.use_supervised_attention:
            spatial_attn = self.spatial_attention(drr)  # [B, 1, 512, 512]
        else:
            spatial_attn = self.spatial_attention(drr, xray_feat)  # [B, 1, 512, 512]
        
        # Resize attention to feature map size
        spatial_attn_resized = F.interpolate(
            spatial_attn, 
            size=adapted_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 16, 16]
        
        # Apply attention modulation with stronger influence
        modulated_feat = adapted_feat * (1 + self.alpha * spatial_attn_resized)
        
        # Generate segmentation
        seg_out = self.segmentation_head(modulated_feat)
        
        # Prepare output
        output = {
            'segmentation': seg_out,
            'attention': spatial_attn,
            'nodule_features': nodule_specific_feat
        }
        
        # Add attention loss if training with supervision
        if ground_truth_mask is not None and self.use_supervised_attention:
            attention_loss = self.spatial_attention.attention_loss(spatial_attn, ground_truth_mask)
            output['attention_loss'] = attention_loss
        
        return output

# Backward compatibility wrapper
class SmallDatasetXrayDRRModel(ImprovedXrayDRRModel):
    """Updated model with improved attention mechanism."""
    
    def __init__(self, pretrained_model=None, alpha=0.3, freeze_early_layers=True, target_pathology='Nodule'):
        super().__init__(
            pretrained_model=pretrained_model,
            alpha=alpha,
            use_supervised_attention=True,  # Enable supervised attention by default
            target_pathology=target_pathology
        )
        print(f"Using improved model with supervised attention (alpha={alpha})")
