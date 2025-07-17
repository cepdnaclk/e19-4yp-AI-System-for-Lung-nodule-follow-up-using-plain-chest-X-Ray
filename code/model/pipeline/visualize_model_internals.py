"""
Comprehensive visualization of model internals for nodule-specific lightweight model.
Shows feature maps, attention, fusion, and segmentation outputs.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

# Try to import skimage for image resizing, fallback to torch interpolation if not available
try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. Using PyTorch interpolation for resizing.")

from small_dataset_config import SmallDatasetConfig
from improved_model import ImprovedXrayDRRModel  # Use the improved model
from improved_config import IMPROVED_CONFIG  # Use improved config
from drr_dataset_loading import DRRDataset


class ModelInternalVisualizer:
    """Visualize internal model representations and intermediate outputs."""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or SmallDatasetConfig()
        self.device = self.config.DEVICE
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            print("Creating new model for visualization...")
            self.model = ImprovedXrayDRRModel(
                pretrained_model=None,
                alpha=IMPROVED_CONFIG['ALPHA'],
                use_supervised_attention=IMPROVED_CONFIG['USE_SUPERVISED_ATTENTION'],
                target_pathology=IMPROVED_CONFIG['TARGET_PATHOLOGY']
            ).to(self.device)
        
        self.model.eval()
        
        # Create hooks to capture intermediate features
        self.feature_maps = {}
        self._register_hooks()
        
    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        print(f"Loading model from {model_path}")
        try:
            # Try loading with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Creating new model instead...")
            return ImprovedXrayDRRModel(
                pretrained_model=None,
                alpha=IMPROVED_CONFIG['ALPHA'],
                use_supervised_attention=IMPROVED_CONFIG['USE_SUPERVISED_ATTENTION'],
                target_pathology=IMPROVED_CONFIG['TARGET_PATHOLOGY']
            ).to(self.device)
        
        model = ImprovedXrayDRRModel(
            pretrained_model=None,
            alpha=IMPROVED_CONFIG['ALPHA'],
            use_supervised_attention=IMPROVED_CONFIG['USE_SUPERVISED_ATTENTION'],
            target_pathology=IMPROVED_CONFIG['TARGET_PATHOLOGY']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate feature maps."""
        
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for key layers (improved model architecture)
        self.model.feature_extractor[-1].register_forward_hook(get_activation('backbone_features'))
        self.model.nodule_feature_extractor.register_forward_hook(get_activation('nodule_specific_features'))
        self.model.nodule_adaptation.register_forward_hook(get_activation('adapted_features'))
        # Note: feature_fusion doesn't exist in ImprovedXrayDRRModel, removed this hook
        self.model.spatial_attention.register_forward_hook(get_activation('spatial_attention'))
        self.model.segmentation_head.register_forward_hook(get_activation('segmentation_output'))
    
    def visualize_sample(self, sample_idx=0, save_dir='visualizations', from_training=False):
        """Visualize a single sample through the entire pipeline."""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load dataset (training or validation)
        if from_training:
            dataset = DRRDataset(
                data_root=self.config.DATA_ROOT,
                image_size=self.config.IMAGE_SIZE,
                training=True,
                augment=False,  # No augmentation for visualization
                normalize=self.config.NORMALIZE_DATA
            )
            dataset_type = "training"
        else:
            dataset = DRRDataset(
                data_root=self.config.DATA_ROOT,
                image_size=self.config.IMAGE_SIZE,
                training=False,
                augment=False,
                normalize=self.config.NORMALIZE_DATA
            )
            dataset_type = "validation"
        
        print(f"Dataset: {dataset_type} set with {len(dataset)} samples")
        
        if sample_idx >= len(dataset):
            raise ValueError(f"Sample index {sample_idx} is out of range. {dataset_type.capitalize()} set has {len(dataset)} samples (0-{len(dataset)-1})")
        
        # Get sample
        xray, drr, mask = dataset[sample_idx]
        xray = xray.unsqueeze(0).to(self.device)  # Add batch dimension
        drr = drr.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        
        print(f"Visualizing sample {sample_idx} from {dataset_type} set")
        print(f"Input shapes - X-ray: {xray.shape}, DRR: {drr.shape}, Mask: {mask.shape}")
        
        # Forward pass
        with torch.no_grad():
            result = self.model(xray, drr)
            segmentation = result['segmentation']
            attention = result['attention']
            nodule_features = result['nodule_features']
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(
            xray, drr, mask, segmentation, attention, 
            save_dir, sample_idx, dataset_type
        )
        
        # Create feature map visualizations
        self._visualize_feature_maps(save_dir, sample_idx, dataset_type)
        
        # Create attention analysis
        self._visualize_attention_analysis(xray, drr, attention, save_dir, sample_idx, dataset_type)
        
        print(f"Visualizations saved to {save_dir}/")
    
    def _create_comprehensive_visualization(self, xray, drr, mask, segmentation, attention, save_dir, sample_idx, dataset_type="validation"):
        """Create main overview visualization with fused image."""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Model Pipeline Overview with Fusion - Sample {sample_idx} ({dataset_type} set)', fontsize=16)
        
        # Convert tensors to numpy for visualization
        xray_np = self._tensor_to_numpy(xray)
        drr_np = self._tensor_to_numpy(drr)
        mask_np = self._tensor_to_numpy(mask)
        seg_np = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
        attention_np = attention.cpu().numpy()[0, 0]
        
        # Create fused images using different fusion techniques
        fused_weighted = self._create_weighted_fusion(xray_np, drr_np, attention_np)
        fused_overlay = self._create_overlay_fusion(xray_np, drr_np, attention_np)
        fused_rgb = self._create_rgb_fusion(xray_np, drr_np, attention_np)
        
        # Row 1: Inputs and Attention
        axes[0, 0].imshow(xray_np, cmap='gray')
        axes[0, 0].set_title('Input X-ray')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(drr_np, cmap='gray')
        axes[0, 1].set_title('Input DRR')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(attention_np, cmap='hot')
        axes[0, 2].set_title('Spatial Attention Map')
        axes[0, 2].axis('off')
        
        # Row 2: Fusion Methods
        axes[1, 0].imshow(fused_weighted, cmap='viridis')
        axes[1, 0].set_title('Weighted Fusion\n(X-ray + Attention*DRR)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fused_overlay, cmap='plasma')
        axes[1, 1].set_title('Overlay Fusion\n(High Attention Regions)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(fused_rgb)
        axes[1, 2].set_title('RGB Fusion\n(R:X-ray, G:DRR, B:Attn)')
        axes[1, 2].axis('off')
        
        # Row 3: Features, Prediction and Comparison
        # Show nodule-specific features (average across channels)
        if 'nodule_specific_features' in self.feature_maps:
            nodule_feat = self.feature_maps['nodule_specific_features'].cpu().numpy()[0]
            nodule_feat_avg = np.mean(nodule_feat, axis=0)
            axes[2, 0].imshow(nodule_feat_avg, cmap='viridis')
            axes[2, 0].set_title('Nodule-Specific Features\n(Channel Average)')
            axes[2, 0].axis('off')
        else:
            axes[2, 0].text(0.5, 0.5, 'Nodule Features\nNot Available', 
                           ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].axis('off')
        
        # Show prediction overlay
        axes[2, 1].imshow(xray_np, cmap='gray')
        axes[2, 1].imshow(seg_np, cmap='jet', alpha=0.6)
        axes[2, 1].set_title('Predicted Segmentation\n(on X-ray)')
        axes[2, 1].axis('off')
        
        # Show ground truth vs prediction comparison
        comparison = self._create_comparison_image(mask_np, seg_np)
        axes[2, 2].imshow(comparison)
        axes[2, 2].set_title('GT (Red) vs Pred (Green)\nOverlap (Yellow)')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'overview_sample_{sample_idx}_{dataset_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate detailed fusion analysis
        self._create_detailed_fusion_analysis(xray_np, drr_np, attention_np, mask_np, seg_np, 
                                            save_dir, sample_idx, dataset_type)
    
    def _normalize_image(self, img):
        """Normalize image to [0, 1] range."""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return img
    
    def _create_weighted_fusion(self, xray, drr, attention):
        """Create weighted fusion of X-ray and DRR using attention."""
        # Normalize inputs
        xray_norm = self._normalize_image(xray)
        drr_norm = self._normalize_image(drr)
        attention_norm = self._normalize_image(attention)
        
        # Resize attention to match image dimensions if needed
        if attention_norm.shape != xray_norm.shape:
            if HAS_SKIMAGE:
                attention_norm = resize(attention_norm, xray_norm.shape, preserve_range=True)
            else:
                # Use PyTorch interpolation as fallback
                attention_tensor = torch.from_numpy(attention_norm).unsqueeze(0).unsqueeze(0).float()
                attention_resized = F.interpolate(
                    attention_tensor, 
                    size=xray_norm.shape, 
                    mode='bilinear', 
                    align_corners=False
                )
                attention_norm = attention_resized.squeeze().numpy()
        
        # Weighted combination: X-ray as base, DRR weighted by attention
        alpha = 0.7  # X-ray weight
        beta = 0.3   # DRR weight
        return alpha * xray_norm + beta * drr_norm * attention_norm
    
    def _create_overlay_fusion(self, xray, drr, attention):
        """Create overlay fusion showing high attention regions."""
        # Normalize inputs
        xray_norm = self._normalize_image(xray)
        drr_norm = self._normalize_image(drr)
        attention_norm = self._normalize_image(attention)
        
        # Resize attention if needed
        if attention_norm.shape != xray_norm.shape:
            if HAS_SKIMAGE:
                attention_norm = resize(attention_norm, xray_norm.shape, preserve_range=True)
            else:
                attention_tensor = torch.from_numpy(attention_norm).unsqueeze(0).unsqueeze(0).float()
                attention_resized = F.interpolate(
                    attention_tensor, 
                    size=xray_norm.shape, 
                    mode='bilinear', 
                    align_corners=False
                )
                attention_norm = attention_resized.squeeze().numpy()
        
        # High attention regions get enhanced DRR, low attention regions get X-ray
        threshold = 0.5
        mask = attention_norm > threshold
        fused = xray_norm.copy()
        fused[mask] = 0.5 * xray_norm[mask] + 0.5 * drr_norm[mask]
        return fused
    
    def _create_rgb_fusion(self, xray, drr, attention):
        """Create RGB fusion for multi-channel visualization."""
        # Normalize inputs
        xray_norm = self._normalize_image(xray)
        drr_norm = self._normalize_image(drr)
        attention_norm = self._normalize_image(attention)
        
        # Resize attention if needed
        if attention_norm.shape != xray_norm.shape:
            if HAS_SKIMAGE:
                attention_norm = resize(attention_norm, xray_norm.shape, preserve_range=True)
            else:
                attention_tensor = torch.from_numpy(attention_norm).unsqueeze(0).unsqueeze(0).float()
                attention_resized = F.interpolate(
                    attention_tensor, 
                    size=xray_norm.shape, 
                    mode='bilinear', 
                    align_corners=False
                )
                attention_norm = attention_resized.squeeze().numpy()
        
        # Create RGB image: R=X-ray, G=DRR*attention, B=X-ray*0.5
        h, w = xray_norm.shape
        rgb_fused = np.zeros((h, w, 3))
        rgb_fused[:, :, 0] = xray_norm  # Red channel: X-ray
        rgb_fused[:, :, 1] = drr_norm * attention_norm  # Green channel: attention-weighted DRR
        rgb_fused[:, :, 2] = xray_norm * 0.5  # Blue channel: dimmed X-ray
        return rgb_fused
    
    def _create_detailed_fusion_analysis(self, xray, drr, attention, mask, prediction, save_dir, sample_idx, dataset_type):
        """Create detailed fusion analysis with multiple methods."""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle(f'Detailed Fusion Analysis - Sample {sample_idx} ({dataset_type} set)', fontsize=16)
        
        # Create different fusion methods
        fusion_methods = {
            'Weighted (α=0.7)': self._create_weighted_fusion(xray, drr, attention),
            'Overlay (thresh=0.5)': self._create_overlay_fusion(xray, drr, attention),
            'Multiplicative': self._normalize_image(xray) * self._normalize_image(drr) * self._normalize_image(attention),
            'Additive': 0.5 * (self._normalize_image(xray) + self._normalize_image(drr) * self._normalize_image(attention))
        }
        
        # Top row: Original inputs and fusion methods
        axes[0, 0].imshow(xray, cmap='gray')
        axes[0, 0].set_title('X-ray Input')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(drr, cmap='gray')
        axes[0, 1].set_title('DRR Input')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(attention, cmap='hot')
        axes[0, 2].set_title('Attention Map')
        axes[0, 2].axis('off')
        
        # RGB fusion
        rgb_fusion = self._create_rgb_fusion(xray, drr, attention)
        axes[0, 3].imshow(rgb_fusion)
        axes[0, 3].set_title('RGB Fusion\n(R:X-ray, G:DRR×Attn, B:X-ray×0.5)')
        axes[0, 3].axis('off')
        
        # Bottom row: Different fusion methods
        for i, (method_name, fused_img) in enumerate(fusion_methods.items()):
            axes[1, i].imshow(fused_img, cmap='viridis')
            axes[1, i].set_title(f'{method_name}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'fusion_analysis_sample_{sample_idx}_{dataset_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create fusion quality analysis
        self._create_fusion_quality_analysis(xray, drr, attention, mask, prediction, 
                                           save_dir, sample_idx, dataset_type)
    
    def _create_fusion_quality_analysis(self, xray, drr, attention, mask, prediction, save_dir, sample_idx, dataset_type):
        """Analyze fusion quality and correlation with predictions."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Fusion Quality Analysis - Sample {sample_idx} ({dataset_type} set)', fontsize=16)
        
        # Normalize inputs
        xray_norm = self._normalize_image(xray)
        drr_norm = self._normalize_image(drr)
        attention_norm = self._normalize_image(attention)
        
        # Resize attention if needed
        if attention_norm.shape != xray_norm.shape:
            if HAS_SKIMAGE:
                attention_norm = resize(attention_norm, xray_norm.shape, preserve_range=True)
            else:
                attention_tensor = torch.from_numpy(attention_norm).unsqueeze(0).unsqueeze(0).float()
                attention_resized = F.interpolate(
                    attention_tensor, 
                    size=xray_norm.shape, 
                    mode='bilinear', 
                    align_corners=False
                )
                attention_norm = attention_resized.squeeze().numpy()
        
        # Create different fusion methods
        weighted_fusion = self._create_weighted_fusion(xray, drr, attention)
        
        # Top row: Fusion overlays with ground truth and prediction
        axes[0, 0].imshow(weighted_fusion, cmap='viridis')
        axes[0, 0].contour(mask, colors='red', linewidths=2, alpha=0.8)
        axes[0, 0].set_title('Weighted Fusion + GT Contour (Red)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(weighted_fusion, cmap='viridis')
        axes[0, 1].contour(prediction, colors='yellow', linewidths=2, alpha=0.8)
        axes[0, 1].set_title('Weighted Fusion + Prediction Contour (Yellow)')
        axes[0, 1].axis('off')
        
        # Fusion with both contours
        axes[0, 2].imshow(weighted_fusion, cmap='viridis')
        axes[0, 2].contour(mask, colors='red', linewidths=2, alpha=0.8)
        axes[0, 2].contour(prediction, colors='yellow', linewidths=2, alpha=0.8)
        axes[0, 2].set_title('Fusion + GT (Red) + Pred (Yellow)')
        axes[0, 2].axis('off')
        
        # Bottom row: Attention analysis
        # Attention statistics
        attention_stats = {
            'Mean': np.mean(attention_norm),
            'Std': np.std(attention_norm),
            'Max': np.max(attention_norm),
            'Min': np.min(attention_norm)
        }
        
        # Attention overlay on fusion
        axes[1, 0].imshow(weighted_fusion, cmap='gray')
        axes[1, 0].imshow(attention_norm, cmap='hot', alpha=0.6)
        axes[1, 0].set_title('Fusion + Attention Overlay')
        axes[1, 0].axis('off')
        
        # Attention histogram
        axes[1, 1].hist(attention_norm.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(attention_stats['Mean'], color='red', linestyle='--', 
                          label=f"Mean: {attention_stats['Mean']:.3f}")
        axes[1, 1].set_title('Attention Distribution')
        axes[1, 1].set_xlabel('Attention Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Statistics text
        stats_text = f"""Attention Statistics:
        
Mean: {attention_stats['Mean']:.4f}
Std:  {attention_stats['Std']:.4f}
Max:  {attention_stats['Max']:.4f}
Min:  {attention_stats['Min']:.4f}

Fusion Quality Metrics:
• High attention coverage: {np.sum(attention_norm > 0.7) / attention_norm.size * 100:.1f}%
• Medium attention coverage: {np.sum((attention_norm > 0.3) & (attention_norm <= 0.7)) / attention_norm.size * 100:.1f}%
• Low attention coverage: {np.sum(attention_norm <= 0.3) / attention_norm.size * 100:.1f}%

Image Properties:
• X-ray contrast: {np.std(xray_norm):.4f}
• DRR contrast: {np.std(drr_norm):.4f}
• Fusion contrast: {np.std(weighted_fusion):.4f}"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'fusion_quality_analysis_sample_{sample_idx}_{dataset_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_maps(self, save_dir, sample_idx, dataset_type="validation"):
        """Create detailed feature map visualizations."""
        
        feature_types = [
            ('backbone_features', 'Backbone Features', 'viridis'),
            ('nodule_specific_features', 'Nodule-Specific Features', 'plasma'),
            ('adapted_features', 'Adapted Features', 'inferno')
        ]
        
        for feat_name, title, cmap in feature_types:
            if feat_name not in self.feature_maps:
                continue
                
            features = self.feature_maps[feat_name].cpu().numpy()[0]  # [C, H, W]
            
            # Select interesting channels to visualize
            num_channels = min(16, features.shape[0])
            channel_indices = np.linspace(0, features.shape[0]-1, num_channels, dtype=int)
            
            # Create subplot grid
            cols = 4
            rows = (num_channels + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
            fig.suptitle(f'{title} - Sample {sample_idx} ({dataset_type} set)', fontsize=14)
            
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, ch_idx in enumerate(channel_indices):
                row = i // cols
                col = i % cols
                
                im = axes[row, col].imshow(features[ch_idx], cmap=cmap)
                axes[row, col].set_title(f'Channel {ch_idx}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(num_channels, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{feat_name}_sample_{sample_idx}_{dataset_type}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_attention_analysis(self, xray, drr, attention, save_dir, sample_idx, dataset_type="validation"):
        """Create detailed attention analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Analysis - Sample {sample_idx} ({dataset_type} set)', fontsize=14)
        
        xray_np = self._tensor_to_numpy(xray)
        drr_np = self._tensor_to_numpy(drr)
        attention_np = attention.cpu().numpy()[0, 0]
        
        # Raw attention
        im1 = axes[0, 0].imshow(attention_np, cmap='hot')
        axes[0, 0].set_title('Raw Attention Map')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Attention overlay on DRR
        axes[0, 1].imshow(drr_np, cmap='gray')
        axes[0, 1].imshow(attention_np, cmap='hot', alpha=0.5)
        axes[0, 1].set_title('Attention on DRR')
        axes[0, 1].axis('off')
        
        # Attention overlay on X-ray
        axes[0, 2].imshow(xray_np, cmap='gray')
        # Resize attention using torch interpolation
        attention_tensor = torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0)
        attention_resized_tensor = F.interpolate(
            attention_tensor, 
            size=(xray_np.shape[0], xray_np.shape[1]), 
            mode='bilinear', 
            align_corners=False
        )
        attention_resized = attention_resized_tensor.squeeze().numpy()
        axes[0, 2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[0, 2].set_title('Attention on X-ray')
        axes[0, 2].axis('off')
        
        # Attention statistics
        axes[1, 0].hist(attention_np.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Attention Value Distribution')
        axes[1, 0].set_xlabel('Attention Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Attention heatmap with contours
        axes[1, 1].imshow(attention_np, cmap='hot')
        contours = axes[1, 1].contour(attention_np, levels=5, colors='white', linewidths=1)
        axes[1, 1].clabel(contours, inline=True, fontsize=8)
        axes[1, 1].set_title('Attention with Contours')
        axes[1, 1].axis('off')
        
        # High attention regions
        threshold = np.percentile(attention_np, 90)
        high_attention = attention_np > threshold
        axes[1, 2].imshow(xray_np, cmap='gray')
        axes[1, 2].imshow(high_attention, cmap='Reds', alpha=0.7)
        axes[1, 2].set_title(f'High Attention Regions\n(Top 10%)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attention_analysis_sample_{sample_idx}_{dataset_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array for visualization."""
        if tensor.dim() == 4:  # [B, C, H, W]
            return tensor.cpu().numpy()[0, 0]
        elif tensor.dim() == 3:  # [C, H, W]
            return tensor.cpu().numpy()[0]
        else:
            return tensor.cpu().numpy()
    
    def _create_comparison_image(self, gt_mask, pred_mask):
        """Create RGB comparison image: GT in red, prediction in green."""
        h, w = gt_mask.shape
        comparison = np.zeros((h, w, 3))
        
        # Ground truth in red channel
        comparison[:, :, 0] = gt_mask
        
        # Prediction in green channel
        comparison[:, :, 1] = pred_mask
        
        # Overlap will appear yellow
        return comparison
    
    def compare_multiple_samples(self, num_samples=5, save_dir='visualizations'):
        """Compare multiple samples side by side."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load dataset
        val_dataset = DRRDataset(
            data_root=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=self.config.NORMALIZE_DATA
        )
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        fig.suptitle('Multi-Sample Comparison', fontsize=16)
        
        for i in range(num_samples):
            # Get sample
            xray, drr, mask = val_dataset[i]
            xray = xray.unsqueeze(0).to(self.device)
            drr = drr.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                result = self.model(xray, drr)
                segmentation = result['segmentation']
                attention = result['attention']
            
            # Convert to numpy
            xray_np = self._tensor_to_numpy(xray)
            mask_np = self._tensor_to_numpy(mask)
            seg_np = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            attention_np = attention.cpu().numpy()[0, 0]
            
            # Plot
            axes[0, i].imshow(xray_np, cmap='gray')
            axes[0, i].set_title(f'Sample {i} - X-ray')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(attention_np, cmap='hot')
            axes[1, i].set_title('Attention')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(mask_np, cmap='jet')
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(seg_np, cmap='jet')
            axes[3, i].set_title('Prediction')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'multi_sample_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-sample comparison saved to {save_dir}/multi_sample_comparison.png")
    
    def compare_multiple_samples_range(self, start_idx=0, num_samples=5, save_dir='visualizations'):
        """Compare multiple samples starting from a specific index."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load dataset
        val_dataset = DRRDataset(
            data_root=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=self.config.NORMALIZE_DATA
        )
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        fig.suptitle(f'Sample Comparison: {start_idx} to {start_idx + num_samples - 1}', fontsize=16)
        
        for i in range(num_samples):
            sample_idx = start_idx + i
            if sample_idx >= len(val_dataset):
                # Hide remaining subplots if we run out of samples
                for row in range(4):
                    axes[row, i].axis('off')
                continue
            
            # Get sample
            xray, drr, mask = val_dataset[sample_idx]
            xray = xray.unsqueeze(0).to(self.device)
            drr = drr.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                result = self.model(xray, drr)
                segmentation = result['segmentation']
                attention = result['attention']
            
            # Convert to numpy
            xray_np = self._tensor_to_numpy(xray)
            mask_np = self._tensor_to_numpy(mask)
            seg_np = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            attention_np = attention.cpu().numpy()[0, 0]
            
            # Plot
            axes[0, i].imshow(xray_np, cmap='gray')
            axes[0, i].set_title(f'Sample {sample_idx} - X-ray')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(attention_np, cmap='hot')
            axes[1, i].set_title('Attention')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(mask_np, cmap='jet')
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(seg_np, cmap='jet')
            axes[3, i].set_title('Prediction')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_samples_{start_idx}_to_{start_idx + num_samples - 1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample range comparison saved to {save_dir}/comparison_samples_{start_idx}_to_{start_idx + num_samples - 1}.png")
    
    def compare_multiple_samples_range_dataset(self, start_idx=0, num_samples=5, save_dir='visualizations', from_training=False):
        """Compare multiple samples starting from a specific index from chosen dataset."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load specified dataset
        dataset = DRRDataset(
            data_root=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            training=from_training,
            augment=False,
            normalize=self.config.NORMALIZE_DATA
        )
        
        dataset_type = "training" if from_training else "validation"
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        fig.suptitle(f'Sample Comparison ({dataset_type} set): {start_idx} to {start_idx + num_samples - 1}', fontsize=16)
        
        for i in range(num_samples):
            sample_idx = start_idx + i
            if sample_idx >= len(dataset):
                # Hide remaining subplots if we run out of samples
                for row in range(4):
                    axes[row, i].axis('off')
                continue
            
            # Get sample
            xray, drr, mask = dataset[sample_idx]
            xray = xray.unsqueeze(0).to(self.device)
            drr = drr.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                result = self.model(xray, drr)
                segmentation = result['segmentation']
                attention = result['attention']
            
            # Convert to numpy
            xray_np = self._tensor_to_numpy(xray)
            mask_np = self._tensor_to_numpy(mask)
            seg_np = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
            attention_np = attention.cpu().numpy()[0, 0]
            
            # Plot
            axes[0, i].imshow(xray_np, cmap='gray')
            axes[0, i].set_title(f'Sample {sample_idx} ({dataset_type}) - X-ray')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(attention_np, cmap='hot')
            axes[1, i].set_title('Attention')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(mask_np, cmap='jet')
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(seg_np, cmap='jet')
            axes[3, i].set_title('Prediction')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{dataset_type}_samples_{start_idx}_to_{start_idx + num_samples - 1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample range comparison ({dataset_type}) saved to {save_dir}/comparison_{dataset_type}_samples_{start_idx}_to_{start_idx + num_samples - 1}.png")


def main():
    """Main visualization function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize model internals for specific samples')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize (default: 0)')
    parser.add_argument('--samples', type=str, default=None, help='Comma-separated list of sample indices (e.g., "0,1,2,3")')
    parser.add_argument('--from-training', action='store_true', help='Use training set instead of validation set')
    parser.add_argument('--multi', type=int, default=5, help='Number of samples for multi-sample comparison (default: 5)')
    parser.add_argument('--output-dir', type=str, default='model_visualizations', help='Output directory for visualizations')
    args = parser.parse_args()
    
    config = SmallDatasetConfig()
    
    # Try to find a trained model - prioritize improved models
    model_paths = [
        'checkpoints_improved/best_model_improved.pth',  # Your latest improved model
        os.path.join(config.SAVE_DIR, 'best_lightweight_model.pth'),
        os.path.join(config.SAVE_DIR, 'final_lightweight_model.pth')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("No trained model found. Creating new model for visualization.")
    
    # Create visualizer
    visualizer = ModelInternalVisualizer(model_path, config)
    
    # Create visualizations directory
    vis_dir = args.output_dir
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Creating comprehensive visualizations...")
    
    # Determine which samples to visualize
    if args.samples:
        # Parse comma-separated list
        sample_indices = [int(x.strip()) for x in args.samples.split(',')]
        print(f"Visualizing specific samples: {sample_indices}")
    else:
        # Use single sample or default range
        sample_indices = [args.sample]
        print(f"Visualizing sample: {args.sample}")
    
    dataset_type = "training" if args.from_training else "validation"
    print(f"Using {dataset_type} set")
    
    # Visualize specified samples
    for i in sample_indices:
        try:
            print(f"\nProcessing sample {i} from {dataset_type} set...")
            visualizer.visualize_sample(sample_idx=i, save_dir=vis_dir, from_training=args.from_training)
            print(f"✓ Sample {i} visualization completed")
        except Exception as e:
            print(f"✗ Error visualizing sample {i}: {e}")
    
    # Create multi-sample comparison
    try:
        print(f"\nCreating multi-sample comparison with {args.multi} samples...")
        visualizer.compare_multiple_samples(num_samples=args.multi, save_dir=vis_dir)
        print("✓ Multi-sample comparison completed")
    except Exception as e:
        print(f"✗ Error creating multi-sample comparison: {e}")
    
    print(f"\nAll visualizations saved to {vis_dir}/")
    print("\nGenerated files:")
    for file in sorted(os.listdir(vis_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")


def visualize_specific_sample(sample_idx, output_dir='model_visualizations', from_training=False):
    """Convenience function to visualize a specific sample."""
    config = SmallDatasetConfig()
    
    # Try to find a trained model - prioritize improved models
    model_paths = [
        'checkpoints_improved/best_model_improved.pth',  # Your latest improved model
        os.path.join(config.SAVE_DIR, 'best_lightweight_model.pth'),
        os.path.join(config.SAVE_DIR, 'final_lightweight_model.pth')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    dataset_type = "training" if from_training else "validation"
    print(f"Visualizing sample {sample_idx} from {dataset_type} set...")
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("No trained model found. Creating new model for visualization.")
    
    # Create visualizer and process sample
    visualizer = ModelInternalVisualizer(model_path, config)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        visualizer.visualize_sample(sample_idx=sample_idx, save_dir=output_dir, from_training=from_training)
        print(f"✓ Sample {sample_idx} ({dataset_type}) visualization saved to {output_dir}/")
        
        # List generated files
        files = [f for f in os.listdir(output_dir) if f.endswith('.png') and f'sample_{sample_idx}_{dataset_type}' in f]
        print(f"Generated {len(files)} visualization files:")
        for file in sorted(files):
            print(f"  - {file}")
            
    except Exception as e:
        print(f"✗ Error visualizing sample {sample_idx}: {e}")
        raise


if __name__ == "__main__":
    main()
