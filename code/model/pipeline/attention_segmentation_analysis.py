"""
Diagnostic tool to analyze attention-segmentation alignment issues.
This helps understand why attention maps might focus on correct areas
but segmentation outputs are misaligned.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Add the current directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightweight_model import SmallDatasetXrayDRRModel
from drr_dataset_loading import DRRDataset
import torchxrayvision as xrv

class AttentionSegmentationAnalyzer:
    """Analyzes the relationship between attention maps and segmentation outputs."""
    
    def __init__(self, model_path=None):
        """Initialize the analyzer with a trained model."""
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = SmallDatasetXrayDRRModel(alpha=0.1)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("Using randomly initialized model for analysis")
        
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Load dataset
        self.dataset = DRRDataset(
            data_root=r"e:\Campus2\fyp-repo\e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray\DRR dataset\LIDC_LDRI",
            training=True,
            augment=False,
            normalize=True
        )
        
    def analyze_sample(self, sample_idx=0, from_training=True):
        """Analyze a specific sample to understand attention-segmentation relationship."""
        
        # Get sample
        if from_training:
            total_samples = len(self.dataset)
            train_size = int(0.8 * total_samples)
            if sample_idx >= train_size:
                sample_idx = 0
                print(f"Sample index too high, using sample 0")
            actual_idx = sample_idx
        else:
            total_samples = len(self.dataset)
            train_size = int(0.8 * total_samples)
            actual_idx = train_size + sample_idx
            if actual_idx >= total_samples:
                actual_idx = train_size
                print(f"Sample index too high, using first validation sample")
        
        sample = self.dataset[actual_idx]
        xray = sample[0].unsqueeze(0).to(self.device)  # Using DRR as xray
        drr = sample[1].unsqueeze(0).to(self.device)
        mask = sample[2].unsqueeze(0).to(self.device)
        
        print(f"Analyzing sample {actual_idx}")
        print(f"X-ray shape: {xray.shape}")
        print(f"DRR shape: {drr.shape}")
        print(f"Mask shape: {mask.shape}")
        
        # Forward pass with intermediate outputs
        with torch.no_grad():
            results = self._forward_with_analysis(xray, drr)
        
        # Analyze results
        analysis = self._analyze_alignment(results, mask, sample_idx, from_training)
        
        return analysis
    
    def _forward_with_analysis(self, xray, drr):
        """Forward pass that captures all intermediate results for analysis."""
        
        results = {}
        
        # 1. Extract X-ray features
        xray_feat = self.model.feature_extractor(xray)
        results['xray_features'] = xray_feat
        print(f"X-ray features shape: {xray_feat.shape}")
        
        # 2. Extract nodule-specific features
        nodule_specific_feat = self.model.nodule_feature_extractor(xray_feat)
        results['nodule_specific_features'] = nodule_specific_feat
        print(f"Nodule-specific features shape: {nodule_specific_feat.shape}")
        
        # 3. Adapt general features to nodule-specific
        adapted_feat = self.model.nodule_adaptation(xray_feat)
        results['adapted_features'] = adapted_feat
        print(f"Adapted features shape: {adapted_feat.shape}")
        
        # 4. Generate spatial attention from DRR
        spatial_attn = self.model.spatial_attention(drr)
        results['attention_full'] = spatial_attn
        print(f"Full attention shape: {spatial_attn.shape}")
        print(f"Attention range: {spatial_attn.min().item():.6f} to {spatial_attn.max().item():.6f}")
        
        # 5. Resize attention to feature map size
        spatial_attn_resized = F.interpolate(
            spatial_attn, 
            size=adapted_feat.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        results['attention_resized'] = spatial_attn_resized
        print(f"Resized attention shape: {spatial_attn_resized.shape}")
        
        # 6. Combine features
        nodule_specific_resized = F.interpolate(
            nodule_specific_feat,
            size=adapted_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        combined_feat = torch.cat([adapted_feat, nodule_specific_resized], dim=1)
        nodule_feat = self.model.feature_fusion(combined_feat)
        results['fused_features'] = nodule_feat
        print(f"Fused features shape: {nodule_feat.shape}")
        
        # 7. Apply attention modulation
        modulated_feat = nodule_feat * (1 + self.model.alpha * spatial_attn_resized)
        results['modulated_features'] = modulated_feat
        print(f"Modulated features shape: {modulated_feat.shape}")
        print(f"Alpha value: {self.model.alpha}")
        
        # 8. Generate segmentation
        seg_out = self.model.segmentation_head(modulated_feat)
        results['segmentation'] = seg_out
        print(f"Segmentation shape: {seg_out.shape}")
        
        return results
    
    def _analyze_alignment(self, results, gt_mask, sample_idx, from_training):
        """Analyze alignment between attention and segmentation."""
        
        analysis = {}
        
        # Convert to numpy for analysis
        attention_full = results['attention_full'].cpu().numpy()[0, 0]  # [512, 512]
        attention_resized = results['attention_resized'].cpu().numpy()[0, 0]  # [16, 16]
        segmentation = torch.sigmoid(results['segmentation']).cpu().numpy()[0, 0]  # [512, 512]
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]  # [512, 512]
        
        # 1. Attention statistics
        analysis['attention_stats'] = {
            'full_range': (attention_full.min(), attention_full.max()),
            'full_mean': attention_full.mean(),
            'full_std': attention_full.std(),
            'resized_range': (attention_resized.min(), attention_resized.max()),
            'resized_mean': attention_resized.mean(),
            'resized_std': attention_resized.std()
        }
        
        # 2. Find attention peaks (top 10% values)
        attention_threshold = np.percentile(attention_full, 90)
        attention_peaks = attention_full > attention_threshold
        
        # 3. Find segmentation peaks (top 10% values)
        seg_threshold = np.percentile(segmentation, 90)
        seg_peaks = segmentation > seg_threshold
        
        # 4. Find GT mask regions
        gt_peaks = gt_mask_np > 0.5
        
        # 5. Calculate overlaps
        attention_gt_overlap = np.sum(attention_peaks & gt_peaks) / np.sum(gt_peaks) if np.sum(gt_peaks) > 0 else 0
        seg_gt_overlap = np.sum(seg_peaks & gt_peaks) / np.sum(gt_peaks) if np.sum(gt_peaks) > 0 else 0
        attention_seg_overlap = np.sum(attention_peaks & seg_peaks) / np.sum(attention_peaks) if np.sum(attention_peaks) > 0 else 0
        
        analysis['overlaps'] = {
            'attention_gt': attention_gt_overlap,
            'segmentation_gt': seg_gt_overlap,
            'attention_segmentation': attention_seg_overlap
        }
        
        # 6. Spatial correlation analysis
        # Flatten arrays for correlation
        attention_flat = attention_full.flatten()
        seg_flat = segmentation.flatten()
        gt_flat = gt_mask_np.flatten()
        
        # Calculate correlations
        attention_gt_corr = np.corrcoef(attention_flat, gt_flat)[0, 1] if np.std(gt_flat) > 0 else 0
        seg_gt_corr = np.corrcoef(seg_flat, gt_flat)[0, 1] if np.std(gt_flat) > 0 else 0
        attention_seg_corr = np.corrcoef(attention_flat, seg_flat)[0, 1]
        
        analysis['correlations'] = {
            'attention_gt': attention_gt_corr,
            'segmentation_gt': seg_gt_corr,
            'attention_segmentation': attention_seg_corr
        }
        
        # 7. Center of mass analysis
        def center_of_mass(img):
            y_coords, x_coords = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
            total_mass = np.sum(img)
            if total_mass > 0:
                y_cm = np.sum(y_coords * img) / total_mass
                x_cm = np.sum(x_coords * img) / total_mass
                return (y_cm, x_cm)
            return (0, 0)
        
        attention_cm = center_of_mass(attention_full)
        seg_cm = center_of_mass(segmentation)
        gt_cm = center_of_mass(gt_mask_np)
        
        analysis['center_of_mass'] = {
            'attention': attention_cm,
            'segmentation': seg_cm,
            'ground_truth': gt_cm,
            'attention_gt_distance': np.sqrt((attention_cm[0] - gt_cm[0])**2 + (attention_cm[1] - gt_cm[1])**2),
            'seg_gt_distance': np.sqrt((seg_cm[0] - gt_cm[0])**2 + (seg_cm[1] - gt_cm[1])**2)
        }
        
        # 8. Save detailed visualization
        self._save_detailed_analysis(results, gt_mask, analysis, sample_idx, from_training)
        
        return analysis
    
    def _save_detailed_analysis(self, results, gt_mask, analysis, sample_idx, from_training):
        """Save detailed visualization of the analysis."""
        
        dataset_type = "training" if from_training else "validation"
        
        # Convert tensors to numpy
        attention_full = results['attention_full'].cpu().numpy()[0, 0]
        attention_resized = results['attention_resized'].cpu().numpy()[0, 0]
        segmentation = torch.sigmoid(results['segmentation']).cpu().numpy()[0, 0]
        gt_mask_np = gt_mask.cpu().numpy()[0, 0]
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Attention-Segmentation Analysis: Sample {sample_idx} ({dataset_type})', fontsize=16)
        
        # Row 1: Original inputs and outputs
        axes[0, 0].imshow(attention_full, cmap='hot')
        axes[0, 0].set_title('Full Attention Map')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(segmentation, cmap='gray')
        axes[0, 1].set_title('Segmentation Output')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(gt_mask_np, cmap='gray')
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        # Overlay attention on segmentation
        axes[0, 3].imshow(segmentation, cmap='gray', alpha=0.7)
        axes[0, 3].imshow(attention_full, cmap='hot', alpha=0.3)
        axes[0, 3].set_title('Attention + Segmentation Overlay')
        axes[0, 3].axis('off')
        
        # Row 2: Peak regions analysis
        attention_threshold = np.percentile(attention_full, 90)
        attention_peaks = attention_full > attention_threshold
        
        seg_threshold = np.percentile(segmentation, 90)
        seg_peaks = segmentation > seg_threshold
        
        gt_peaks = gt_mask_np > 0.5
        
        axes[1, 0].imshow(attention_peaks, cmap='Reds')
        axes[1, 0].set_title('Attention Peaks (Top 10%)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(seg_peaks, cmap='Blues')
        axes[1, 1].set_title('Segmentation Peaks (Top 10%)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(gt_peaks, cmap='Greens')
        axes[1, 2].set_title('Ground Truth Regions')
        axes[1, 2].axis('off')
        
        # Overlap visualization
        overlap_img = np.zeros((*attention_peaks.shape, 3))
        overlap_img[attention_peaks, 0] = 1  # Red for attention
        overlap_img[seg_peaks, 1] = 1        # Green for segmentation
        overlap_img[gt_peaks, 2] = 1         # Blue for GT
        
        axes[1, 3].imshow(overlap_img)
        axes[1, 3].set_title('Overlap Analysis (R:Att, G:Seg, B:GT)')
        axes[1, 3].axis('off')
        
        # Row 3: Analysis metrics and center of mass
        # Text analysis
        axes[2, 0].text(0.05, 0.95, f"Attention Stats:\nRange: {analysis['attention_stats']['full_range'][0]:.4f} - {analysis['attention_stats']['full_range'][1]:.4f}\nMean: {analysis['attention_stats']['full_mean']:.4f}\nStd: {analysis['attention_stats']['full_std']:.4f}", 
                       transform=axes[2, 0].transAxes, fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 0].set_title('Attention Statistics')
        axes[2, 0].axis('off')
        
        axes[2, 1].text(0.05, 0.95, f"Overlaps:\nAtt-GT: {analysis['overlaps']['attention_gt']:.3f}\nSeg-GT: {analysis['overlaps']['segmentation_gt']:.3f}\nAtt-Seg: {analysis['overlaps']['attention_segmentation']:.3f}", 
                       transform=axes[2, 1].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[2, 1].set_title('Overlap Analysis')
        axes[2, 1].axis('off')
        
        axes[2, 2].text(0.05, 0.95, f"Correlations:\nAtt-GT: {analysis['correlations']['attention_gt']:.3f}\nSeg-GT: {analysis['correlations']['segmentation_gt']:.3f}\nAtt-Seg: {analysis['correlations']['attention_segmentation']:.3f}", 
                       transform=axes[2, 2].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        axes[2, 2].set_title('Correlation Analysis')
        axes[2, 2].axis('off')
        
        # Center of mass visualization
        axes[2, 3].imshow(gt_mask_np, cmap='gray', alpha=0.5)
        
        # Plot centers of mass
        att_cm = analysis['center_of_mass']['attention']
        seg_cm = analysis['center_of_mass']['segmentation']
        gt_cm = analysis['center_of_mass']['ground_truth']
        
        axes[2, 3].plot(att_cm[1], att_cm[0], 'ro', markersize=10, label='Attention CM')
        axes[2, 3].plot(seg_cm[1], seg_cm[0], 'bo', markersize=10, label='Segmentation CM')
        axes[2, 3].plot(gt_cm[1], gt_cm[0], 'go', markersize=10, label='GT CM')
        
        axes[2, 3].set_title('Center of Mass Comparison')
        axes[2, 3].legend()
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        # Save the analysis
        output_path = f'attention_analysis_sample_{sample_idx}_{dataset_type}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed analysis saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"ANALYSIS SUMMARY - Sample {sample_idx} ({dataset_type})")
        print("="*60)
        print(f"Attention-GT Overlap: {analysis['overlaps']['attention_gt']:.3f}")
        print(f"Segmentation-GT Overlap: {analysis['overlaps']['segmentation_gt']:.3f}")
        print(f"Attention-Segmentation Overlap: {analysis['overlaps']['attention_segmentation']:.3f}")
        print(f"Attention-GT Correlation: {analysis['correlations']['attention_gt']:.3f}")
        print(f"Segmentation-GT Correlation: {analysis['correlations']['segmentation_gt']:.3f}")
        print(f"Attention-Segmentation Correlation: {analysis['correlations']['attention_segmentation']:.3f}")
        print(f"Attention-GT Distance: {analysis['center_of_mass']['attention_gt_distance']:.1f} pixels")
        print(f"Segmentation-GT Distance: {analysis['center_of_mass']['seg_gt_distance']:.1f} pixels")
        print("="*60)

def main():
    """Run the attention-segmentation analysis."""
    
    # Initialize analyzer
    print("Initializing Attention-Segmentation Analyzer...")
    
    # Look for trained model
    model_paths = [
        'checkpoints/best_model.pth',
        'logs/best_model.pth',
        'best_model.pth'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    analyzer = AttentionSegmentationAnalyzer(model_path)
    
    # Analyze a training sample
    print("\nAnalyzing training sample...")
    analysis = analyzer.analyze_sample(sample_idx=0, from_training=True)
    
    print("\nAnalysis complete! Check the generated visualization files.")
    
    # Additional analysis suggestions
    print("\n" + "="*60)
    print("DIAGNOSTIC SUGGESTIONS")
    print("="*60)
    
    if analysis['correlations']['attention_segmentation'] < 0.3:
        print("⚠️  LOW ATTENTION-SEGMENTATION CORRELATION")
        print("   - The attention map and segmentation are poorly aligned")
        print("   - Consider increasing alpha value or improving attention fusion")
    
    if analysis['overlaps']['attention_gt'] > 0.5 and analysis['overlaps']['segmentation_gt'] < 0.3:
        print("⚠️  ATTENTION FINDS TARGET BUT SEGMENTATION MISSES")
        print("   - Attention mechanism is working but segmentation head is failing")
        print("   - Check segmentation head architecture and training")
    
    if analysis['overlaps']['attention_gt'] < 0.3:
        print("⚠️  ATTENTION NOT FOCUSING ON TARGET")
        print("   - The attention mechanism itself needs improvement")
        print("   - Consider using supervised attention or different DRR preprocessing")
    
    if analysis['center_of_mass']['attention_gt_distance'] > 100:
        print("⚠️  LARGE SPATIAL MISALIGNMENT")
        print("   - Attention center is far from ground truth center")
        print("   - Check data preprocessing and spatial alignment")

if __name__ == "__main__":
    main()
