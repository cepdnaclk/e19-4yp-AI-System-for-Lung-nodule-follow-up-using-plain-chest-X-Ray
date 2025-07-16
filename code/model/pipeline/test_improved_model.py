"""
Test script for the improved model to validate it addresses the attention-segmentation alignment issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import our modules
from improved_model import ImprovedXrayDRRModel
from drr_dataset_loading import DRRDataset

def test_improved_model():
    """Test the improved model with attention analysis."""
    
    print("Testing Improved Model with Supervised Attention")
    print("="*60)
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize improved model
    print("\nInitializing improved model...")
    model = ImprovedXrayDRRModel(alpha=0.3, use_supervised_attention=True)
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Load a sample from the dataset
    print("\nLoading test sample...")
    
    # Use cross-platform path detection
    data_root = os.path.join("..", "..", "..", "DRR dataset", "LIDC_LDRI")
    
    # If relative path doesn't exist, try absolute path detection
    if not os.path.exists(data_root):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        data_root = os.path.join(project_root, "DRR dataset", "LIDC_LDRI")
        
        if not os.path.exists(data_root):
            # Last resort: look for any LIDC_LDRI directory
            for root, dirs, files in os.walk(os.path.dirname(project_root)):
                if "LIDC_LDRI" in dirs:
                    data_root = os.path.join(root, "LIDC_LDRI")
                    break
    
    print(f"Using dataset path: {data_root}")
    
    dataset = DRRDataset(
        data_root=data_root,
        training=True,
        augment=False,
        normalize=True
    )
    
    # Get a sample
    sample_idx = 0
    xray, drr, mask = dataset[sample_idx]
    
    # Add batch dimension
    xray = xray.unsqueeze(0).to(device)
    drr = drr.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    
    print(f"Sample shapes - X-ray: {xray.shape}, DRR: {drr.shape}, Mask: {mask.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        # Test without supervision
        results_unsupervised = model(xray, drr)
        print("✓ Unsupervised forward pass successful")
        
        # Test with supervision (training mode)
        results_supervised = model(xray, drr, ground_truth_mask=mask)
        print("✓ Supervised forward pass successful")
        
        # Check outputs
        print(f"\nOutput analysis:")
        print(f"Segmentation shape: {results_supervised['segmentation'].shape}")
        print(f"Attention shape: {results_supervised['attention'].shape}")
        print(f"Nodule features shape: {results_supervised['nodule_features'].shape}")
        
        # Analyze attention quality
        attention = results_supervised['attention'].cpu().numpy()[0, 0]
        segmentation = torch.sigmoid(results_supervised['segmentation']).cpu().numpy()[0, 0]
        ground_truth = mask.cpu().numpy()[0, 0]
        
        print(f"\nAttention analysis:")
        print(f"Attention range: {attention.min():.4f} to {attention.max():.4f}")
        print(f"Attention mean: {attention.mean():.4f}")
        print(f"Attention std: {attention.std():.4f}")
        
        # Check if attention supervision loss is available
        if 'attention_loss' in results_supervised:
            att_loss = results_supervised['attention_loss']
            print(f"Attention supervision loss: {att_loss.item():.4f}")
        else:
            print("No attention supervision loss (normal for inference)")
        
        # Calculate basic overlap metrics
        attention_threshold = np.percentile(attention, 90)
        attention_peaks = attention > attention_threshold
        gt_peaks = ground_truth > 0.5
        
        if np.sum(gt_peaks) > 0:
            attention_gt_overlap = np.sum(attention_peaks & gt_peaks) / np.sum(gt_peaks)
            print(f"Attention-GT overlap: {attention_gt_overlap:.3f}")
        else:
            print("No ground truth regions found")
        
        # Save visualization
        save_improved_visualization(xray, drr, attention, segmentation, ground_truth, sample_idx)
    
    print("\n" + "="*60)
    print("IMPROVED MODEL TEST COMPLETED!")
    print("="*60)
    print("Key improvements:")
    print("1. ✓ Supervised attention mechanism")
    print("2. ✓ Anatomically-aware attention fusion")
    print("3. ✓ Better attention normalization")
    print("4. ✓ Attention supervision loss")
    print("5. ✓ Improved feature fusion")
    print("\nNext steps:")
    print("1. Run: python train_improved_simple.py")
    print("2. Compare attention maps with original model")
    print("3. Evaluate attention-segmentation alignment")

def save_improved_visualization(xray, drr, attention, segmentation, ground_truth, sample_idx):
    """Save visualization of improved model outputs."""
    
    # Convert tensors to numpy
    xray_img = xray.cpu().numpy()[0, 0]
    drr_img = drr.cpu().numpy()[0, 0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Improved Model Analysis - Sample {sample_idx}', fontsize=16)
    
    # Top row: inputs and attention
    axes[0, 0].imshow(xray_img, cmap='gray')
    axes[0, 0].set_title('X-ray Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(drr_img, cmap='gray')
    axes[0, 1].set_title('DRR Input')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(attention, cmap='hot')
    axes[0, 2].set_title(f'Improved Attention\\n(Range: {attention.min():.3f}-{attention.max():.3f})')
    axes[0, 2].axis('off')
    
    # Bottom row: outputs and comparison
    axes[1, 0].imshow(segmentation, cmap='gray')
    axes[1, 0].set_title('Segmentation Output')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ground_truth, cmap='gray')
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].axis('off')
    
    # Combined overlay
    axes[1, 2].imshow(ground_truth, cmap='Greens', alpha=0.4, label='GT')
    axes[1, 2].imshow(segmentation, cmap='Blues', alpha=0.4, label='Pred')
    axes[1, 2].imshow(attention, cmap='Reds', alpha=0.3, label='Att')
    axes[1, 2].set_title('Overlay (G:GT, B:Pred, R:Att)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = f'improved_model_test_sample_{sample_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    test_improved_model()
