"""
Lightweight model architecture specifically designed for small datasets (~600 samples).
Focuses on regularization and simplicity to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightAttentionModule(nn.Module):
    """Lightweight attention module designed for small datasets."""
    
    def __init__(self, in_channels=1, hidden_channels=16):  # Much smaller capacity
        super().__init__()
        
        # Simple but effective attention network
        self.attention_net = nn.Sequential(
            # First layer with heavy dropout
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Heavy regularization
            
            # Second layer
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # Output layer
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, drr):
        return self.attention_net(drr)

class SimpleUpsamplingHead(nn.Module):
    """Simple upsampling head optimized for small datasets."""
    
    def __init__(self, in_channels=128, hidden_channels=64):  # Updated default
        super().__init__()
        
        # Very simple decoder to prevent overfitting
        self.decoder = nn.Sequential(
            # Initial channel reduction with heavy regularization
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),  # Heavy dropout
            
            # Simple upsampling layers
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16->32
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32->64
            nn.Conv2d(hidden_channels//2, hidden_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64->128
            nn.Conv2d(hidden_channels//4, hidden_channels//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 128->512
            nn.Conv2d(hidden_channels//8, 1, kernel_size=1)
            # Removed Sigmoid - let loss function handle it for numerical stability
        )
        
    def forward(self, x):
        return self.decoder(x)

class SmallDatasetXrayDRRModel(nn.Module):
    """Lightweight model specifically designed for small datasets (~600 samples)."""
    
    def __init__(self, pretrained_model=None, alpha=0.2, freeze_early_layers=True, target_pathology='Nodule'):
        super().__init__()
        self.alpha = alpha
        self.target_pathology = target_pathology
        
        # Initialize pretrained model if not provided
        if pretrained_model is None:
            import torchxrayvision as xrv
            pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        
        self.pretrained_model = pretrained_model
        
        # Get the pathology index for nodule-specific features
        self.pathology_idx = None
        if hasattr(pretrained_model, 'pathologies'):
            try:
                self.pathology_idx = pretrained_model.pathologies.index(target_pathology)
                print(f"Found {target_pathology} at index {self.pathology_idx}")
            except ValueError:
                # If exact match not found, look for similar terms
                similar_pathologies = [p for p in pretrained_model.pathologies 
                                     if target_pathology.lower() in p.lower() or 
                                        'mass' in p.lower() or 'lesion' in p.lower()]
                if similar_pathologies:
                    self.pathology_idx = pretrained_model.pathologies.index(similar_pathologies[0])
                    print(f"Using similar pathology: {similar_pathologies[0]} at index {self.pathology_idx}")
                else:
                    print(f"Warning: {target_pathology} not found in pathologies: {pretrained_model.pathologies}")
        
        # Feature extractor - heavily frozen to prevent overfitting
        # Remove the classification head (avgpool, fc) since we only need features
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
        
        if freeze_early_layers:
            # Freeze ALL backbone layers except the very last one
            for param in self.feature_extractor[:-1].parameters():
                param.requires_grad = False
            
            # Only allow minimal adaptation in the last layer
            for param in self.feature_extractor[-1].parameters():
                param.requires_grad = True
        
        # CRITICAL: Nodule-specific feature adaptation using pretrained pathology knowledge
        # Extract nodule-specific features from the classification weights
        self.nodule_feature_extractor = self._create_nodule_feature_extractor()
        
        # Additional adaptation layers for segmentation
        self.nodule_adaptation = nn.Sequential(
            # Transform classification features to nodule-specific features
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Further refinement for nodule characteristics
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Final nodule-specific feature representation
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def _create_nodule_feature_extractor(self):
        """Create a nodule-specific feature extractor using pretrained pathology weights."""
        # Get the classification weights from the pretrained model
        if hasattr(self.pretrained_model, 'classifier') and self.pathology_idx is not None:
            # Extract the weights corresponding to nodule classification
            classifier_weights = self.pretrained_model.classifier.weight.data
            nodule_weights = classifier_weights[self.pathology_idx:self.pathology_idx+1]  # Shape: [1, 2048]
            
            # Create a 1x1 conv layer initialized with nodule-specific weights
            nodule_conv = nn.Conv2d(2048, 1, kernel_size=1, bias=False)
            nodule_conv.weight.data = nodule_weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [1, 2048, 1, 1]
            
            # Create feature enhancement layer
            enhancement = nn.Sequential(
                nodule_conv,
                nn.Sigmoid(),  # Attention-like weights
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            # Freeze the nodule-specific conv to preserve pretrained knowledge
            nodule_conv.weight.requires_grad = False
            
            return enhancement
        else:
            # Fallback: simple feature enhancement without pretrained weights
            return nn.Sequential(
                nn.Conv2d(2048, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3)
            )
        
        # Lightweight attention module
        self.spatial_attention = LightweightAttentionModule(in_channels=1, hidden_channels=16)
        
        # Feature fusion layer for combining nodule-specific and adapted features
        self.feature_fusion = nn.Conv2d(192, 128, kernel_size=1)  # 128 + 64 = 192
        
        # Simple upsampling head - updated to work with adapted features
        self.segmentation_head = SimpleUpsamplingHead(in_channels=128, hidden_channels=64)
        
        # Optional feature fusion (very simple)
        self.use_fusion = False  # Disable complex fusion for small datasets
        
    def forward(self, xray, drr):
        # Extract features from X-ray (general pathology features)
        xray_feat = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # CRITICAL: Extract nodule-specific features using pretrained pathology knowledge
        nodule_specific_feat = self.nodule_feature_extractor(xray_feat)  # [B, 64, 16, 16]
        
        # Further adapt general pathology features to nodule-specific features
        adapted_feat = self.nodule_adaptation(xray_feat)  # [B, 128, 16, 16]
        
        # Combine nodule-specific and adapted features
        # Resize nodule-specific features to match adapted features (should already match)
        nodule_specific_resized = F.interpolate(
            nodule_specific_feat,
            size=adapted_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Concatenate and reduce dimensions
        combined_feat = torch.cat([adapted_feat, nodule_specific_resized], dim=1)  # [B, 192, 16, 16]
        
        # Final feature fusion
        nodule_feat = self.feature_fusion(combined_feat)  # [B, 128, 16, 16]
        
        # Generate spatial attention from DRR
        spatial_attn = self.spatial_attention(drr)  # [B, 1, 512, 512]
        
        # Resize attention to feature map size
        spatial_attn_resized = F.interpolate(
            spatial_attn, 
            size=nodule_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )  # [B, 1, 16, 16]
        
        # Simple attention modulation on nodule-specific features
        modulated_feat = nodule_feat * (1 + self.alpha * spatial_attn_resized)
        
        # Generate segmentation
        seg_out = self.segmentation_head(modulated_feat)
        
        return {
            'segmentation': seg_out,
            'attention': spatial_attn,
            'nodule_features': nodule_specific_feat  # For visualization/debugging
        }

# Additional lightweight variant for extremely small datasets
class MinimalXrayDRRModel(nn.Module):
    """Minimal model for very small datasets with extreme regularization."""
    
    def __init__(self, pretrained_model=None, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
        # Initialize pretrained model if not provided
        if pretrained_model is None:
            import torchxrayvision as xrv
            pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        
        # Use only up to layer3 to reduce complexity
        self.feature_extractor = nn.Sequential(
            pretrained_model.model.conv1,
            pretrained_model.model.bn1,
            pretrained_model.model.relu,
            pretrained_model.model.maxpool,
            pretrained_model.model.layer1,
            pretrained_model.model.layer2,
            pretrained_model.model.layer3,  # Stop here (1024 channels instead of 2048)
        )
        
        # Freeze everything
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Nodule-specific adaptation for 1024-channel features
        self.nodule_adaptation = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Minimal attention
        self.attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Minimal decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # Input from nodule adaptation
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 32->128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 128->512
            nn.Conv2d(16, 1, kernel_size=1)
            # No sigmoid - let loss handle it
        )
        
    def forward(self, xray, drr):
        # Extract features (smaller feature map: 32x32 instead of 16x16)
        xray_feat = self.feature_extractor(xray)  # [B, 1024, 32, 32]
        
        # Adapt to nodule-specific features
        nodule_feat = self.nodule_adaptation(xray_feat)  # [B, 64, 32, 32]
        
        # Simple attention
        attn = self.attention(drr)  # [B, 1, 512, 512]
        attn_resized = F.interpolate(attn, size=nodule_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Simple modulation
        modulated = nodule_feat * (1 + self.alpha * attn_resized)
        
        # Decode
        output = self.decoder(modulated)
        
        return {
            'segmentation': output,
            'attention': attn
        }

# Backward compatibility
class XrayDRRSegmentationModel(SmallDatasetXrayDRRModel):
    """Backward compatibility wrapper using the lightweight model."""
    
    def __init__(self, pretrained_model=None, alpha=0.2, target_pathology='Nodule'):
        super().__init__(pretrained_model, alpha, freeze_early_layers=True, target_pathology=target_pathology)
        # Keep old naming for compatibility
        self.attention_net = self.spatial_attention
        
    def forward(self, xray, drr):
        result = super().forward(xray, drr)
        # Return only segmentation for old compatibility
        return result['segmentation']
