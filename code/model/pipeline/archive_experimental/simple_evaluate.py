"""
Simple evaluation script for the improved model.
Clean and focused on essential metrics.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import torchxrayvision as xrv

from drr_dataset_loading import DRRSegmentationDataset
from improved_model import ImprovedXrayDRRSegmentationModel
from improved_utils import calculate_metrics, find_optimal_threshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path, config=None):
    """Evaluate the trained model with clean, simple metrics."""
    
    # Default config
    if config is None:
        config = {
            'data_root': '../../../DRR dataset/LIDC_LDRI',
            'image_size': (512, 512),
            'batch_size': 1,  # Single image evaluation
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
    
    logger.info(f"Using device: {config['device']}")
    
    # Load dataset
    dataset = DRRSegmentationDataset(
        root_dir=config['data_root'],
        image_size=config['image_size'],
        augment=False  # No augmentation for evaluation
    )
    
    # Create test loader
    test_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2
    )
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=config['device'])
        
        # Initialize model
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(
            pretrained_model, 
            alpha=checkpoint.get('config', {}).get('alpha', 0.3)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config['device'])
        model.eval()
        
        optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        logger.info(f"Model loaded. Optimal threshold: {optimal_threshold:.3f}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Evaluation
    all_preds = []
    all_masks = []
    sample_count = 0
    max_samples = 100  # Limit for faster evaluation
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= max_samples:
                break
                
            try:
                xray = batch["xray"].to(config['device'])
                drr = batch["drr"].to(config['device'])
                mask = batch["mask"].to(config['device'])
                
                # Forward pass
                seg_out, attn_map = model(xray, drr)
                
                all_preds.append(seg_out.cpu())
                all_masks.append(mask.cpu())
                sample_count += 1
                
                if batch_idx % 20 == 0:
                    logger.info(f"Processed {batch_idx + 1} samples...")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
    
    if not all_preds:
        logger.error("No predictions collected!")
        return
    
    # Combine all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    logger.info(f"Evaluation completed on {len(all_preds)} samples")
    
    # Calculate metrics with default threshold
    default_metrics = calculate_metrics(all_preds, all_masks, threshold=0.5)
    
    # Calculate metrics with optimal threshold
    optimal_metrics = calculate_metrics(all_preds, all_masks, threshold=optimal_threshold)
    
    # Find best threshold on this test set
    test_optimal_threshold, test_best_dice = find_optimal_threshold(all_preds, all_masks)
    test_optimal_metrics = calculate_metrics(all_preds, all_masks, threshold=test_optimal_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nDefault Threshold (0.5):")
    print(f"  Dice:      {default_metrics['dice']:.4f}")
    print(f"  IoU:       {default_metrics['iou']:.4f}")
    print(f"  Precision: {default_metrics['precision']:.4f}")
    print(f"  Recall:    {default_metrics['recall']:.4f}")
    print(f"  F1:        {default_metrics['f1']:.4f}")
    
    print(f"\nOptimal Threshold ({optimal_threshold:.3f}):")
    print(f"  Dice:      {optimal_metrics['dice']:.4f}")
    print(f"  IoU:       {optimal_metrics['iou']:.4f}")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall:    {optimal_metrics['recall']:.4f}")
    print(f"  F1:        {optimal_metrics['f1']:.4f}")
    
    print(f"\nTest Set Optimal Threshold ({test_optimal_threshold:.3f}):")
    print(f"  Dice:      {test_optimal_metrics['dice']:.4f}")
    print(f"  IoU:       {test_optimal_metrics['iou']:.4f}")
    print(f"  Precision: {test_optimal_metrics['precision']:.4f}")
    print(f"  Recall:    {test_optimal_metrics['recall']:.4f}")
    print(f"  F1:        {test_optimal_metrics['f1']:.4f}")
    
    # Visualize some results
    visualize_results(
        all_preds[:8], all_masks[:8], 
        threshold=test_optimal_threshold,
        save_path='evaluation_results.png'
    )
    
    return {
        'default': default_metrics,
        'optimal': optimal_metrics,
        'test_optimal': test_optimal_metrics,
        'optimal_threshold': optimal_threshold,
        'test_optimal_threshold': test_optimal_threshold
    }

def visualize_results(preds, masks, threshold=0.5, save_path=None):
    """Visualize prediction results."""
    num_samples = min(8, len(preds))
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        pred = preds[i][0].numpy()
        mask = masks[i][0].numpy()
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Calculate metrics for this sample
        sample_metrics = calculate_metrics(
            torch.tensor(pred).unsqueeze(0).unsqueeze(0),
            torch.tensor(mask).unsqueeze(0).unsqueeze(0),
            threshold
        )
        
        # Original prediction (probability)
        axes[0, i].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f'Prediction {i+1}')
        axes[0, i].axis('off')
        
        # Binary prediction
        axes[1, i].imshow(pred_binary, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'Binary (t={threshold:.2f})')
        axes[1, i].axis('off')
        
        # Ground truth
        axes[2, i].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'GT (Dice: {sample_metrics["dice"]:.3f})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    model_path = "./checkpoints/best_model.pth"
    if os.path.exists(model_path):
        results = evaluate_model(model_path)
    else:
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train the model first using improved_train.py")
