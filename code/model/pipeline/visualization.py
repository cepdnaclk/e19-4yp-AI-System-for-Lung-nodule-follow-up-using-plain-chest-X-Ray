"""
Advanced visualization tools for debugging and understanding the model behavior.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns
from matplotlib.patches import Rectangle
import logging

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Comprehensive visualization tool for model debugging."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def visualize_attention_mechanism(self, xray, drr, mask, save_path=None):
        """Visualize how the attention mechanism works."""
        with torch.no_grad():
            # Get model outputs - improved compatibility
            try:
                result = self.model(xray, drr)
                
                if isinstance(result, dict):
                    # New ImprovedXrayDRRSegmentationModel
                    pred_mask = result['segmentation']
                    attention_map = result.get('attention', None)
                    
                    if attention_map is None:
                        # Fallback: generate attention separately
                        attention_map = self.model.spatial_attention(drr)
                else:
                    # Old model or backward compatibility
                    pred_mask = result
                    
                    # Try different attribute names for attention
                    if hasattr(self.model, 'spatial_attention'):
                        attention_map = self.model.spatial_attention(drr)
                    elif hasattr(self.model, 'attention_net'):
                        attention_map = self.model.attention_net(drr)
                    else:
                        # Create dummy attention map if none available
                        attention_map = torch.ones_like(drr) * 0.5
                        print("⚠️ Warning: No attention mechanism found, using dummy attention map")
                        
            except Exception as e:
                print(f"❌ Error getting model outputs: {e}")
                return
        
        # Convert to numpy
        xray_np = xray[0, 0].cpu().numpy()
        drr_np = drr[0, 0].cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_mask[0, 0].cpu().numpy()
        attn_np = attention_map[0, 0].cpu().numpy()
        
        # Create visualization with improved layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original X-ray
        axes[0, 0].imshow(xray_np, cmap='gray')
        axes[0, 0].set_title('Input X-ray', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # DRR input
        axes[0, 1].imshow(drr_np, cmap='gray')
        axes[0, 1].set_title('DRR Input (Attention Source)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Attention map with colorbar
        im_attn = axes[0, 2].imshow(attn_np, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Generated Attention Map', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im_attn, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Ground truth
        axes[1, 0].imshow(mask_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Prediction with threshold line
        im_pred = axes[1, 1].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Model Prediction (Raw)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im_pred, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Overlay attention on X-ray with better blending
        overlay = np.stack([xray_np, xray_np, xray_np], axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        attn_colored = plt.cm.hot(attn_np)[:, :, :3]
        # Better blending: more X-ray, less attention for clarity
        overlay = 0.8 * overlay + 0.2 * attn_colored
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('X-ray + Attention Overlay', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Add metrics as text
        pred_binary = (pred_np > 0.5).astype(float)
        intersection = (pred_binary * mask_np).sum()
        dice = (2 * intersection) / (pred_binary.sum() + mask_np.sum() + 1e-8)
        
        fig.suptitle(f'Attention Mechanism Analysis (Dice Score: {dice:.3f})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def analyze_prediction_quality(self, dataloader, num_samples=20, save_path=None):
        """Analyze prediction quality across multiple samples."""
        self.model.eval()
        samples_analyzed = 0
        
        all_dices = []
        all_ious = []
        all_precisions = []
        all_recalls = []
        
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        axes = axes.flatten()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if samples_analyzed >= num_samples:
                    break
                    
                xray = batch["xray"].to(self.device)
                drr = batch["drr"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                # Get predictions with improved compatibility
                try:
                    result = self.model(xray, drr)
                    if isinstance(result, dict):
                        pred_mask = result['segmentation']
                    else:
                        pred_mask = result
                except Exception as e:
                    logger.warning(f"Error getting predictions for batch {batch_idx}: {e}")
                    continue
                
                for i in range(min(xray.shape[0], num_samples - samples_analyzed)):
                    # Calculate metrics
                    pred_np = pred_mask[i, 0].cpu().numpy()
                    mask_np = mask[i, 0].cpu().numpy()
                    
                    # Binary prediction
                    pred_binary = (pred_np > 0.5).astype(float)
                    
                    # Calculate metrics
                    intersection = (pred_binary * mask_np).sum()
                    union = pred_binary.sum() + mask_np.sum() - intersection
                    dice = (2 * intersection) / (pred_binary.sum() + mask_np.sum() + 1e-7)
                    iou = intersection / (union + 1e-7)
                    
                    tp = intersection
                    fp = pred_binary.sum() - intersection
                    fn = mask_np.sum() - intersection
                    precision = tp / (tp + fp + 1e-7)
                    recall = tp / (tp + fn + 1e-7)
                    
                    all_dices.append(dice)
                    all_ious.append(iou)
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    
                    # Visualize
                    ax = axes[samples_analyzed]
                    
                    # Create composite image
                    xray_np = xray[i, 0].cpu().numpy()
                    composite = np.stack([xray_np, pred_np, mask_np])
                    composite = np.transpose(composite, (1, 2, 0))
                    
                    ax.imshow(composite)
                    ax.set_title(f'Sample {samples_analyzed + 1}\nDice: {dice:.3f}, IoU: {iou:.3f}', 
                               fontsize=10)
                    ax.axis('off')
                    
                    samples_analyzed += 1
                    
                    if samples_analyzed >= num_samples:
                        break
        
        # Remove unused subplots
        for i in range(samples_analyzed, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_quality_analysis.png'), dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close()
        
        # Print statistics
        print(f"\nPrediction Quality Analysis ({samples_analyzed} samples):")
        print(f"Dice Score - Mean: {np.mean(all_dices):.4f}, Std: {np.std(all_dices):.4f}")
        print(f"IoU Score - Mean: {np.mean(all_ious):.4f}, Std: {np.std(all_ious):.4f}")
        print(f"Precision - Mean: {np.mean(all_precisions):.4f}, Std: {np.std(all_precisions):.4f}")
        print(f"Recall - Mean: {np.mean(all_recalls):.4f}, Std: {np.std(all_recalls):.4f}")
        
        return {
            'dice': all_dices,
            'iou': all_ious,
            'precision': all_precisions,
            'recall': all_recalls
        }
    
    def visualize_feature_maps(self, xray, drr, layer_name='layer4', save_path=None):
        """Visualize intermediate feature maps."""
        hooks = []
        feature_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # Register hooks
        if hasattr(self.model, 'feature_extractor'):
            for name, module in self.model.feature_extractor.named_children():
                if layer_name in name:
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(xray, drr)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Visualize feature maps
        for name, features in feature_maps.items():
            feat = features[0]  # First sample in batch
            n_features = min(16, feat.shape[0])  # Show up to 16 feature maps
            
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            
            for i in range(n_features):
                feat_map = feat[i].cpu().numpy()
                axes[i].imshow(feat_map, cmap='viridis')
                axes[i].set_title(f'Feature {i}')
                axes[i].axis('off')
            
            # Remove unused subplots
            for i in range(n_features, 16):
                fig.delaxes(axes[i])
            
            plt.suptitle(f'Feature Maps from {name}', fontsize=16)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path.replace('.png', f'_features_{name}.png'), 
                           dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
    
    def threshold_analysis(self, dataloader, save_path=None):
        """Analyze performance across different thresholds."""
        self.model.eval()
        
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for batch in dataloader:
                xray = batch["xray"].to(self.device)
                drr = batch["drr"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                # Get predictions with improved compatibility
                try:
                    result = self.model(xray, drr)
                    if isinstance(result, dict):
                        pred_mask = result['segmentation']
                    else:
                        pred_mask = result
                except Exception as e:
                    logger.warning(f"Error getting predictions: {e}")
                    continue
                
                all_preds.append(pred_mask.cpu())
                all_masks.append(mask.cpu())
        
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Test different thresholds
        thresholds = np.linspace(0.1, 0.9, 17)
        metrics = {'threshold': [], 'dice': [], 'iou': [], 'precision': [], 'recall': []}
        
        for threshold in thresholds:
            pred_binary = (all_preds > threshold).float()
            
            # Calculate metrics
            intersection = (pred_binary * all_masks).sum()
            union = pred_binary.sum() + all_masks.sum() - intersection
            
            dice = (2 * intersection) / (pred_binary.sum() + all_masks.sum() + 1e-7)
            iou = intersection / (union + 1e-7)
            
            tp = intersection
            fp = pred_binary.sum() - intersection
            fn = all_masks.sum() - intersection
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            
            metrics['threshold'].append(threshold)
            metrics['dice'].append(dice.item())
            metrics['iou'].append(iou.item())
            metrics['precision'].append(precision.item())
            metrics['recall'].append(recall.item())
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(metrics['threshold'], metrics['dice'], 'b-o', linewidth=2)
        axes[0, 0].set_title('Dice Score vs Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Dice Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(metrics['threshold'], metrics['iou'], 'r-o', linewidth=2)
        axes[0, 1].set_title('IoU vs Threshold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(metrics['threshold'], metrics['precision'], 'g-o', linewidth=2)
        axes[1, 0].set_title('Precision vs Threshold')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(metrics['threshold'], metrics['recall'], 'm-o', linewidth=2)
        axes[1, 1].set_title('Recall vs Threshold')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_threshold_analysis.png'), 
                       dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        
        # Find optimal threshold
        best_dice_idx = np.argmax(metrics['dice'])
        optimal_threshold = metrics['threshold'][best_dice_idx]
        best_dice = metrics['dice'][best_dice_idx]
        
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Best Dice Score: {best_dice:.4f}")
        print(f"Corresponding IoU: {metrics['iou'][best_dice_idx]:.4f}")
        print(f"Corresponding Precision: {metrics['precision'][best_dice_idx]:.4f}")
        print(f"Corresponding Recall: {metrics['recall'][best_dice_idx]:.4f}")
        
        return metrics, optimal_threshold

def visualize_dataset_statistics(dataset, save_path=None):
    """Visualize dataset statistics and class distribution."""
    stats = dataset.get_statistics()
    
    # Sample some data points
    sample_masks = []
    sample_images = []
    
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        mask = sample['mask'].numpy()
        image = sample['xray'].numpy()
        
        sample_masks.append(mask.mean())  # Fraction of positive pixels
        sample_images.append(image.mean())  # Average intensity
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mask ratio distribution
    axes[0, 0].hist(sample_masks, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Positive Pixel Ratios')
    axes[0, 0].set_xlabel('Positive Pixel Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Image intensity distribution
    axes[0, 1].hist(sample_images, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Distribution of Image Intensities')
    axes[0, 1].set_xlabel('Average Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample images
    sample_indices = np.random.choice(len(dataset), 8, replace=False)
    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        ax = axes[1, 0] if i < 4 else axes[1, 1]
        row = i % 4
        
        if i == 0:
            ax.set_title('Sample Images and Masks')
        
        # Show image and mask side by side
        img = sample['xray'][0].numpy()
        mask = sample['mask'][0].numpy()
        
        combined = np.concatenate([img, mask], axis=1)
        ax.imshow(combined, cmap='gray', aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
    
    if len(sample_indices) <= 4:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_dataset_stats.png'), 
                   dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print("Dataset Statistics:")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Image size: {stats['image_size']}")
    print(f"Average positive pixel ratio: {np.mean(sample_masks):.4f} ± {np.std(sample_masks):.4f}")
    print(f"Average image intensity: {np.mean(sample_images):.4f} ± {np.std(sample_images):.4f}")
