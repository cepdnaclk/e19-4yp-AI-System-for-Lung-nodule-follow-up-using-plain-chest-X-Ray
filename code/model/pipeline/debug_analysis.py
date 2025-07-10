"""
Quick evaluation and debugging script to understand current model performance.
"""

import torch
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

from drr_dataset_loading import DRRSegmentationDataset
from custom_model import XrayDRRSegmentationModel, ImprovedXrayDRRSegmentationModel
from util import calculate_metrics
from visualization import ModelVisualizer

def quick_debug():
    """Quick debugging function to analyze current model issues."""
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = '../../../DRR dataset/LIDC_LDRI'
    image_size = (512, 512)
    batch_size = 2
    
    print(f"Using device: {device}")
    
    # Load dataset
    try:
        dataset = DRRSegmentationDataset(
            root_dir=data_root,
            image_size=image_size,
            augment=False,
            normalize=True
        )
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Get dataset statistics
        stats = dataset.get_statistics()
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    xray = sample_batch["xray"]
    drr = sample_batch["drr"]
    mask = sample_batch["mask"]
    
    print(f"Sample shapes:")
    print(f"  X-ray: {xray.shape}")
    print(f"  DRR: {drr.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Mask statistics:")
    print(f"    Positive pixels ratio: {mask.mean().item():.6f}")
    print(f"    Min: {mask.min():.3f}, Max: {mask.max():.3f}")
    
    # Load pretrained model
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        print("Pretrained model loaded successfully")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return
    
    # Test both old and new models
    models_to_test = [
        ("Original Model", XrayDRRSegmentationModel(pretrained_model, alpha=0.5)),
        ("Improved Model", ImprovedXrayDRRSegmentationModel(pretrained_model, alpha=0.3))
    ]
    
    results = {}
    
    for model_name, model in models_to_test:
        print(f"\n=== Testing {model_name} ===")
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            xray_gpu = xray.to(device)
            drr_gpu = drr.to(device)
            mask_gpu = mask.to(device)
            
            # Forward pass
            try:
                output = model(xray_gpu, drr_gpu)
                
                if isinstance(output, dict):
                    pred_mask = output['segmentation']
                    attention_map = output.get('attention', None)
                else:
                    pred_mask = output
                    attention_map = None
                
                print(f"Prediction shape: {pred_mask.shape}")
                print(f"Prediction statistics:")
                print(f"  Min: {pred_mask.min():.6f}, Max: {pred_mask.max():.6f}")
                print(f"  Mean: {pred_mask.mean():.6f}, Std: {pred_mask.std():.6f}")
                print(f"  Positive predictions (>0.5): {(pred_mask > 0.5).float().mean():.6f}")
                
                # Calculate metrics for different thresholds
                thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                print(f"Metrics for different thresholds:")
                
                best_dice = 0
                best_threshold = 0.5
                
                for threshold in thresholds:
                    pred_binary = (pred_mask > threshold).float()
                    metrics = calculate_metrics(pred_binary, mask_gpu, threshold=0.5)
                    
                    print(f"  Threshold {threshold}: Dice={metrics['dice']:.4f}, "
                          f"IoU={metrics['iou']:.4f}, Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}")
                    
                    if metrics['dice'] > best_dice:
                        best_dice = metrics['dice']
                        best_threshold = threshold
                
                print(f"Best threshold: {best_threshold} with Dice: {best_dice:.4f}")
                
                results[model_name] = {
                    'predictions': pred_mask.cpu(),
                    'attention': attention_map.cpu() if attention_map is not None else None,
                    'best_threshold': best_threshold,
                    'best_dice': best_dice
                }
                
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
    
    # Visualization
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(len(models_to_test) + 1, 4, figsize=(20, 5 * (len(models_to_test) + 1)))
    
    # First row: Input data
    sample_idx = 0
    xray_np = xray[sample_idx, 0].numpy()
    drr_np = drr[sample_idx, 0].numpy()
    mask_np = mask[sample_idx, 0].numpy()
    
    axes[0, 0].imshow(xray_np, cmap='gray')
    axes[0, 0].set_title('Input X-ray')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(drr_np, cmap='gray')
    axes[0, 1].set_title('Input DRR')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_np, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth Mask')
    axes[0, 2].axis('off')
    
    axes[0, 3].text(0.1, 0.5, f'Dataset Info:\\nSamples: {len(dataset)}\\nPos ratio: {mask.mean():.6f}\\nImage size: {image_size}', 
                   transform=axes[0, 3].transAxes, fontsize=10, verticalalignment='center')
    axes[0, 3].axis('off')
    
    # Model predictions
    for idx, (model_name, result) in enumerate(results.items()):
        row = idx + 1
        pred_np = result['predictions'][sample_idx, 0].numpy()
        pred_binary = (pred_np > result['best_threshold']).astype(float)
        
        # Raw prediction
        axes[row, 0].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        axes[row, 0].set_title(f'{model_name}\\nRaw Prediction')
        axes[row, 0].axis('off')
        
        # Binary prediction with optimal threshold
        axes[row, 1].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Binary (t={result["best_threshold"]:.1f})')
        axes[row, 1].axis('off')
        
        # Attention map (if available)
        if result['attention'] is not None:
            attn_np = result['attention'][sample_idx, 0].numpy()
            axes[row, 2].imshow(attn_np, cmap='hot', vmin=0, vmax=1)
            axes[row, 2].set_title('Attention Map')
        else:
            axes[row, 2].text(0.5, 0.5, 'No Attention\\nMap Available', 
                            transform=axes[row, 2].transAxes, ha='center', va='center')
        axes[row, 2].axis('off')
        
        # Metrics
        dice = calculate_metrics(torch.tensor(pred_binary).unsqueeze(0).unsqueeze(0), 
                               mask[sample_idx:sample_idx+1])['dice']
        axes[row, 3].text(0.1, 0.5, f'Metrics:\\nBest Dice: {result["best_dice"]:.4f}\\nOptimal Threshold: {result["best_threshold"]:.2f}\\nActual Dice: {dice:.4f}', 
                         transform=axes[row, 3].transAxes, fontsize=10, verticalalignment='center')
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Detailed analysis
    print("\n=== Detailed Analysis ===")
    print("Common Issues to Check:")
    print("1. Class imbalance: Very few positive pixels in masks")
    print("2. Model overpredicting: Too many false positives")
    print("3. Attention mechanism: Not focusing on correct regions")
    print("4. Loss function: May need better handling of class imbalance")
    
    print("\nRecommendations:")
    print("1. Use weighted loss functions (higher weight for positive class)")
    print("2. Optimize threshold based on validation data")
    print("3. Add data augmentation to increase positive samples")
    print("4. Consider using focal loss to focus on hard examples")
    print("5. Visualize attention maps to ensure they focus on nodule regions")
    
    return results

if __name__ == "__main__":
    results = quick_debug()
