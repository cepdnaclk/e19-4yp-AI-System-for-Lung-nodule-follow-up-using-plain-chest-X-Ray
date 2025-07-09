"""
Simplified training script focused on accuracy and clean implementation.
Removes unnecessary complexity while maintaining effective training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import torchxrayvision as xrv

from drr_dataset_loading import DRRSegmentationDataset
from improved_model import ImprovedXrayDRRSegmentationModel
from improved_utils import combined_loss, calculate_metrics, find_optimal_threshold

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model():
    # Simplified configuration
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (512, 512),
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'alpha': 0.3,  # Attention modulation factor
        'save_dir': './checkpoints',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    logger.info(f"Using device: {config['device']}")
    
    # Load dataset
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config['data_root'],
            image_size=config['image_size'],
            augment=True  # Enable augmentation for training
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=2, pin_memory=True)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Load pretrained model and initialize our model
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(pretrained_model, alpha=config['alpha'])
        model = model.to(config['device'])
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Optimizer with different learning rates for different parts
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': config['learning_rate'] * 0.1},  # Lower LR for pretrained
        {'params': model.attention_net.parameters(), 'lr': config['learning_rate']},
        {'params': model.decoder.parameters(), 'lr': config['learning_rate']}
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_dice = 0.0
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

                # Main segmentation loss
                loss_seg = combined_loss(seg_out, mask)
                
                # Attention supervision loss (lighter weight)
                attn_resized = F.interpolate(attn_map, size=mask.shape[2:], 
                                           mode='bilinear', align_corners=False)
                loss_attn = F.binary_cross_entropy(attn_resized, mask)
                
                # Total loss
                loss_total = loss_seg + 0.1 * loss_attn  # Reduced attention weight

                loss_total.backward()
                optimizer.step()

                train_loss += loss_total.item()
                train_batches += 1
                
                if batch_idx % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                              f"Batch {batch_idx}, Loss: {loss_total.item():.4f}")
                    
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
                    loss_seg = combined_loss(seg_out, mask)
                    attn_resized = F.interpolate(attn_map, size=mask.shape[2:], 
                                               mode='bilinear', align_corners=False)
                    loss_attn = F.binary_cross_entropy(attn_resized, mask)
                    loss_total = loss_seg + 0.1 * loss_attn

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
                       f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                       f"Val Dice: {metrics['dice']:.4f} (threshold: {optimal_threshold:.3f})")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model based on dice score
        if val_dices and val_dices[-1] > best_val_dice:
            best_val_dice = val_dices[-1]
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'optimal_threshold': optimal_threshold,
                'config': config
            }
            
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model.pth'))
            logger.info(f"New best model saved with dice score: {best_val_dice:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_dices, label='Val Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Show final metrics
    if all_preds is not None and all_masks is not None:
        final_metrics = calculate_metrics(all_preds, all_masks, optimal_threshold)
        metrics_text = f"Final Metrics (threshold={optimal_threshold:.3f}):\n"
        metrics_text += f"Dice: {final_metrics['dice']:.4f}\n"
        metrics_text += f"IoU: {final_metrics['iou']:.4f}\n"
        metrics_text += f"Precision: {final_metrics['precision']:.4f}\n"
        metrics_text += f"Recall: {final_metrics['recall']:.4f}\n"
        metrics_text += f"F1: {final_metrics['f1']:.4f}"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        plt.title('Final Performance')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_results.png'), dpi=150)
    plt.close()
    
    logger.info(f"Training completed. Best validation dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    train_model()
