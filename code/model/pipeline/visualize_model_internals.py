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

from small_dataset_config import SmallDatasetConfig
from lightweight_model import SmallDatasetXrayDRRModel
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
            self.model = SmallDatasetXrayDRRModel(
                pretrained_model=None,
                alpha=self.config.ALPHA,
                freeze_early_layers=True,
                target_pathology='Nodule'
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
            return SmallDatasetXrayDRRModel(
                pretrained_model=None,
                alpha=self.config.ALPHA,
                freeze_early_layers=True,
                target_pathology='Nodule'
            ).to(self.device)
        
        model = SmallDatasetXrayDRRModel(
            pretrained_model=None,
            alpha=self.config.ALPHA,
            freeze_early_layers=True,
            target_pathology='Nodule'
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate feature maps."""
        
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        self.model.feature_extractor[-1].register_forward_hook(get_activation('backbone_features'))
        self.model.nodule_feature_extractor.register_forward_hook(get_activation('nodule_specific_features'))
        self.model.nodule_adaptation.register_forward_hook(get_activation('adapted_features'))
        self.model.feature_fusion.register_forward_hook(get_activation('fused_features'))
        self.model.spatial_attention.register_forward_hook(get_activation('spatial_attention'))
        self.model.segmentation_head.register_forward_hook(get_activation('segmentation_output'))
    
    def visualize_sample(self, sample_idx=0, save_dir='visualizations'):
        """Visualize a single sample through the entire pipeline."""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load dataset
        val_dataset = DRRDataset(
            data_root=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=self.config.NORMALIZE_DATA
        )
        
        # Get sample
        xray, drr, mask = val_dataset[sample_idx]
        xray = xray.unsqueeze(0).to(self.device)  # Add batch dimension
        drr = drr.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        
        print(f"Visualizing sample {sample_idx}")
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
            save_dir, sample_idx
        )
        
        # Create feature map visualizations
        self._visualize_feature_maps(save_dir, sample_idx)
        
        # Create attention analysis
        self._visualize_attention_analysis(xray, drr, attention, save_dir, sample_idx)
        
        print(f"Visualizations saved to {save_dir}/")
    
    def _create_comprehensive_visualization(self, xray, drr, mask, segmentation, attention, save_dir, sample_idx):
        """Create main overview visualization."""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Model Pipeline Overview - Sample {sample_idx}', fontsize=16)
        
        # Convert tensors to numpy for visualization
        xray_np = self._tensor_to_numpy(xray)
        drr_np = self._tensor_to_numpy(drr)
        mask_np = self._tensor_to_numpy(mask)
        seg_np = torch.sigmoid(segmentation).cpu().numpy()[0, 0]
        attention_np = attention.cpu().numpy()[0, 0]
        
        # Row 1: Inputs and Ground Truth
        axes[0, 0].imshow(xray_np, cmap='gray')
        axes[0, 0].set_title('Input X-ray')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(drr_np, cmap='gray')
        axes[0, 1].set_title('Input DRR')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask_np, cmap='jet', alpha=0.7)
        axes[0, 2].imshow(xray_np, cmap='gray', alpha=0.3)
        axes[0, 2].set_title('Ground Truth Mask')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(attention_np, cmap='hot')
        axes[0, 3].set_title('Spatial Attention')
        axes[0, 3].axis('off')
        
        # Row 2: Intermediate and Final Outputs
        # Show nodule-specific features (average across channels)
        if 'nodule_specific_features' in self.feature_maps:
            nodule_feat = self.feature_maps['nodule_specific_features'].cpu().numpy()[0]
            nodule_feat_avg = np.mean(nodule_feat, axis=0)
            axes[1, 0].imshow(nodule_feat_avg, cmap='viridis')
            axes[1, 0].set_title('Nodule-Specific Features\n(Channel Average)')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Nodule Features\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Show fused features
        if 'fused_features' in self.feature_maps:
            fused_feat = self.feature_maps['fused_features'].cpu().numpy()[0]
            fused_feat_avg = np.mean(fused_feat, axis=0)
            axes[1, 1].imshow(fused_feat_avg, cmap='plasma')
            axes[1, 1].set_title('Fused Features\n(Channel Average)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'Fused Features\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        # Show prediction
        axes[1, 2].imshow(seg_np, cmap='jet', alpha=0.7)
        axes[1, 2].imshow(xray_np, cmap='gray', alpha=0.3)
        axes[1, 2].set_title('Predicted Segmentation')
        axes[1, 2].axis('off')
        
        # Show comparison
        comparison = self._create_comparison_image(mask_np, seg_np)
        axes[1, 3].imshow(comparison)
        axes[1, 3].set_title('GT (Red) vs Pred (Green)')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'overview_sample_{sample_idx}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_feature_maps(self, save_dir, sample_idx):
        """Create detailed feature map visualizations."""
        
        feature_types = [
            ('backbone_features', 'Backbone Features', 'viridis'),
            ('nodule_specific_features', 'Nodule-Specific Features', 'plasma'),
            ('adapted_features', 'Adapted Features', 'inferno'),
            ('fused_features', 'Fused Features', 'magma')
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
            fig.suptitle(f'{title} - Sample {sample_idx}', fontsize=14)
            
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
            plt.savefig(os.path.join(save_dir, f'{feat_name}_sample_{sample_idx}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_attention_analysis(self, xray, drr, attention, save_dir, sample_idx):
        """Create detailed attention analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Attention Analysis - Sample {sample_idx}', fontsize=14)
        
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
        plt.savefig(os.path.join(save_dir, f'attention_analysis_sample_{sample_idx}.png'), 
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


def main():
    """Main visualization function."""
    
    config = SmallDatasetConfig()
    
    # Try to find a trained model
    model_paths = [
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
    vis_dir = 'model_visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Creating comprehensive visualizations...")
    
    # Visualize individual samples
    for i in range(3):  # Visualize first 3 samples
        try:
            visualizer.visualize_sample(sample_idx=i, save_dir=vis_dir)
        except Exception as e:
            print(f"Error visualizing sample {i}: {e}")
    
    # Create multi-sample comparison
    try:
        visualizer.compare_multiple_samples(num_samples=5, save_dir=vis_dir)
    except Exception as e:
        print(f"Error creating multi-sample comparison: {e}")
    
    print(f"\nAll visualizations saved to {vis_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(vis_dir):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == "__main__":
    main()
