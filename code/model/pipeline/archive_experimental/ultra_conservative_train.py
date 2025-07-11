"""
Ultra-conservative training specifically for medical segmentation with extreme precision focus.
Designed to drastically reduce false positives.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
import torchxrayvision as xrv
import numpy as np

from drr_dataset_loading import DRRSegmentationDataset
from improved_model import ImprovedXrayDRRSegmentationModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ultra_conservative_loss(pred, target, precision_penalty=10.0):
    """Ultra-conservative loss that heavily penalizes false positives."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # True positives, false positives, false negatives
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    # Precision-focused Tversky loss with extreme false positive penalty
    precision_tversky = (tp + 1e-6) / (tp + precision_penalty * fp + 0.3 * fn + 1e-6)
    
    # Add entropy regularization to prevent overconfident predictions
    entropy_reg = -(pred_flat * torch.log(pred_flat + 1e-8) + 
                   (1 - pred_flat) * torch.log(1 - pred_flat + 1e-8)).mean()
    
    return 1 - precision_tversky + 0.1 * entropy_reg

def calculate_metrics_simple(pred, target, threshold=0.5):
    """Simple metrics calculation."""
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def train_ultra_conservative():
    """Ultra-conservative training to drastically improve precision."""
    
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (256, 256),  # Reduced size for faster training
        'batch_size': 1,  # Single sample for maximum stability
        'learning_rate': 1e-6,  # Extremely conservative LR
        'num_epochs': 50,
        'alpha': 0.05,  # Minimal attention modulation
        'save_dir': './checkpoints_ultra_conservative',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'early_stopping_patience': 15,
        'grad_clip': 0.5,
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    logger.info(f"Using device: {config['device']}")
    logger.info("Starting ultra-conservative training for precision improvement")
    
    # Load dataset with minimal augmentation
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config['data_root'],
            image_size=config['image_size'],
            augment=False  # No augmentation for maximum stability
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Smaller dataset for faster iteration
        if len(dataset) > 200:
            subset_indices = torch.randperm(len(dataset))[:200]
            dataset = torch.utils.data.Subset(dataset, subset_indices)
            logger.info(f"Using subset of {len(dataset)} samples for faster training")
        
        # Train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=0)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Initialize model with ultra-conservative settings
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(pretrained_model, alpha=config['alpha'])
        model = model.to(config['device'])
        
        # Freeze even more layers for maximum conservatism
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        
        # Only train the final layers
        for param in model.decoder.final.parameters():
            param.requires_grad = True
        for param in model.attention_net.parameters():
            param.requires_grad = True
            
        logger.info("Model initialized with ultra-conservative settings (minimal trainable parameters)")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Ultra-conservative optimizer - only train essential parts
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            logger.info(f"Training parameter: {name}")
    
    optimizer = torch.optim.SGD(trainable_params, lr=config['learning_rate'], 
                               momentum=0.9, weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training metrics
    best_precision = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_precisions = []
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

                # Ultra-conservative loss (heavy false positive penalty)
                loss_seg = ultra_conservative_loss(seg_out, mask, precision_penalty=20.0)
                
                # No attention supervision - focus purely on segmentation
                loss_total = loss_seg

                loss_total.backward()
                
                # Aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, config['grad_clip'])
                
                optimizer.step()

                train_loss += loss_total.item()
                train_batches += 1
                
                if batch_idx % 20 == 0:
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
                    loss_seg = ultra_conservative_loss(seg_out, mask, precision_penalty=20.0)
                    loss_total = loss_seg

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
        
        # Calculate validation metrics for multiple thresholds
        if all_preds and all_masks:
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            
            # Test multiple high thresholds to find best precision
            best_threshold = 0.5
            best_precision_score = 0.0
            best_metrics = None
            
            for threshold in [0.7, 0.8, 0.9, 0.95, 0.99]:
                metrics = calculate_metrics_simple(all_preds, all_masks, threshold)
                if metrics['precision'] > best_precision_score:
                    best_precision_score = metrics['precision']
                    best_threshold = threshold
                    best_metrics = metrics
            
            val_precisions.append(best_precision_score)
            val_dices.append(best_metrics['dice'])
            
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                       f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                       f"Best Precision: {best_precision_score:.4f}, Dice: {best_metrics['dice']:.4f}, "
                       f"Recall: {best_metrics['recall']:.4f} (threshold: {best_threshold:.2f})")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save model based on precision improvement
        if val_precisions and val_precisions[-1] > best_precision:
            best_precision = val_precisions[-1]
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_precision': best_precision,
                'best_threshold': best_threshold,
                'config': config,
                'metrics': best_metrics
            }
            
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_precision_model.pth'))
            logger.info(f"New best precision model saved: {best_precision:.4f} at threshold {best_threshold:.2f}")
        else:
            patience_counter += 1
            
        # Early stopping based on precision
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping: no precision improvement for {patience_counter} epochs")
            break
    
    # Final results
    logger.info(f"Training completed. Best precision: {best_precision:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_precisions, 'r-', label='Precision')
    plt.plot(val_dices, 'b-', label='Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if best_metrics:
        metrics_text = f"Best Results (threshold={best_threshold:.2f}):\n\n"
        metrics_text += f"Precision: {best_metrics['precision']:.4f}\n"
        metrics_text += f"Dice: {best_metrics['dice']:.4f}\n"
        metrics_text += f"Recall: {best_metrics['recall']:.4f}\n"
        metrics_text += f"IoU: {best_metrics['iou']:.4f}\n"
        metrics_text += f"F1: {best_metrics['f1']:.4f}\n\n"
        metrics_text += f"Improvement vs baseline:\n"
        metrics_text += f"Precision: {best_metrics['precision']/0.003:.1f}x better"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center')
        plt.title('Final Performance')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'ultra_conservative_results.png'), dpi=150)
    plt.close()
    
    return best_metrics, best_threshold

if __name__ == "__main__":
    print("Starting ultra-conservative training...")
    print("Focus: Drastically improve precision by reducing false positives")
    print("Strategy: Heavy false positive penalty + minimal trainable parameters")
    print("-" * 60)
    
    metrics, threshold = train_ultra_conservative()
    
    print(f"\nULTRA-CONSERVATIVE TRAINING RESULTS:")
    print(f"Precision: {metrics['precision']:.4f} (target: >0.1)")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Optimal Threshold: {threshold:.2f}")
    print(f"Precision Improvement: {metrics['precision']/0.003:.1f}x better than baseline")
