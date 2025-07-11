"""
Aggressive training script specifically designed for extreme class imbalance.
Focused on precision improvement and reduced false positives.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import torchxrayvision as xrv
import numpy as np

from drr_dataset_loading import DRRSegmentationDataset
from improved_model import ImprovedXrayDRRSegmentationModel
from improved_utils import aggressive_combined_loss, calculate_metrics, find_optimal_threshold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_with_class_balance_focus():
    """Training focused on handling extreme class imbalance."""
    
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (512, 512),
        'batch_size': 2,  # Small batch for stability
        'learning_rate': 2e-5,  # Very conservative LR
        'num_epochs': 30,
        'alpha': 0.15,  # Minimal attention modulation
        'save_dir': './checkpoints_aggressive',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'early_stopping_patience': 8,
        'grad_clip': 1.0,  # Gradient clipping
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    logger.info(f"Using device: {config['device']}")
    
    # Load dataset with aggressive augmentation
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config['data_root'],
            image_size=config['image_size'],
            augment=True
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Train/val split with fixed seed
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=1, pin_memory=True)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Initialize model with conservative settings
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(pretrained_model, alpha=config['alpha'])
        model = model.to(config['device'])
        logger.info("Model initialized with aggressive regularization")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Very conservative optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': config['learning_rate'] * 0.01, 'weight_decay': 1e-3},
        {'params': model.attention_net.parameters(), 'lr': config['learning_rate'] * 0.3, 'weight_decay': 5e-4},
        {'params': model.decoder.parameters(), 'lr': config['learning_rate'], 'weight_decay': 5e-4},
        {'params': model.global_fc.parameters(), 'lr': config['learning_rate'], 'weight_decay': 5e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-8
    )
    
    # Training with early stopping
    best_val_dice = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dices = []
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                xray = batch["xray"].to(config['device'])
                drr = batch["drr"].to(config['device'])
                mask = batch["mask"].to(config['device'])

                optimizer.zero_grad()

                # Forward pass
                seg_out, attn_map = model(xray, drr)

                # Aggressive loss for class imbalance
                loss_seg = aggressive_combined_loss(seg_out, mask)
                
                # Minimal attention supervision
                attn_resized = F.interpolate(attn_map, size=mask.shape[2:], 
                                           mode='bilinear', align_corners=False)
                loss_attn = F.binary_cross_entropy(attn_resized, mask)
                
                # Total loss - focus on segmentation
                loss_total = loss_seg + 0.02 * loss_attn

                loss_total.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                optimizer.step()

                train_loss += loss_total.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                              f"Batch {batch_idx}, Loss: {loss_total.item():.6f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    xray = batch["xray"].to(config['device'])
                    drr = batch["drr"].to(config['device'])
                    mask = batch["mask"].to(config['device'])

                    seg_out, attn_map = model(xray, drr)
                    
                    # Calculate loss
                    loss_seg = aggressive_combined_loss(seg_out, mask)
                    attn_resized = F.interpolate(attn_map, size=mask.shape[2:], 
                                               mode='bilinear', align_corners=False)
                    loss_attn = F.binary_cross_entropy(attn_resized, mask)
                    loss_total = loss_seg + 0.02 * loss_attn

                    val_loss += loss_total.item()
                    val_batches += 1
                    
                    # Collect predictions for metrics
                    all_preds.append(seg_out.cpu())
                    all_masks.append(mask.cpu())
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Calculate validation metrics
        if all_preds and all_masks:
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            
            # Find optimal threshold and calculate metrics
            optimal_threshold, best_dice = find_optimal_threshold(all_preds, all_masks)
            metrics = calculate_metrics(all_preds, all_masks, optimal_threshold)
            val_dices.append(metrics['dice'])
            
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                       f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                       f"Val Dice: {metrics['dice']:.4f}, Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f} (threshold: {optimal_threshold:.3f})")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and model saving
        if val_dices and val_dices[-1] > best_val_dice:
            best_val_dice = val_dices[-1]
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'optimal_threshold': optimal_threshold,
                'config': config,
                'metrics': metrics
            }
            
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model_aggressive.pth'))
            logger.info(f"New best model saved with dice: {best_val_dice:.4f}, precision: {metrics['precision']:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Final evaluation and visualization
    logger.info(f"Training completed. Best validation dice: {best_val_dice:.4f}")
    
    # Plot comprehensive results
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 4, 2)
    plt.plot(val_dices, label='Val Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    if all_preds is not None and all_masks is not None:
        final_metrics = calculate_metrics(all_preds, all_masks, optimal_threshold)
        metrics_text = f"Final Metrics (threshold={optimal_threshold:.3f}):\n\n"
        metrics_text += f"Dice: {final_metrics['dice']:.4f}\n"
        metrics_text += f"IoU: {final_metrics['iou']:.4f}\n"
        metrics_text += f"Precision: {final_metrics['precision']:.4f}\n"
        metrics_text += f"Recall: {final_metrics['recall']:.4f}\n"
        metrics_text += f"F1: {final_metrics['f1']:.4f}\n\n"
        metrics_text += f"Improvement over baseline:\n"
        metrics_text += f"Dice: {final_metrics['dice']/0.0085:.1f}x better\n"
        metrics_text += f"Precision: {final_metrics['precision']/0.0043:.1f}x better"
        
        plt.text(0.05, 0.5, metrics_text, fontsize=10, verticalalignment='center')
        plt.title('Final Performance')
        plt.axis('off')
    
    plt.subplot(1, 4, 4)
    # Threshold analysis
    thresholds = np.linspace(0.1, 0.9, 17)
    dice_scores = []
    precision_scores = []
    
    for thresh in thresholds:
        metrics = calculate_metrics(all_preds, all_masks, thresh)
        dice_scores.append(metrics['dice'])
        precision_scores.append(metrics['precision'])
    
    plt.plot(thresholds, dice_scores, 'b-', label='Dice Score')
    plt.plot(thresholds, precision_scores, 'r-', label='Precision')
    plt.axvline(optimal_threshold, color='g', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'aggressive_training_results.png'), dpi=150)
    plt.close()
    
    return final_metrics, optimal_threshold

if __name__ == "__main__":
    metrics, threshold = train_with_class_balance_focus()
    print(f"\nFinal Results:")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Optimal Threshold: {threshold:.3f}")
