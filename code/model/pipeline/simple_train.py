"""
Simple training script for lightweight model on small datasets.
Optimized for preventing overfitting with ~600 samples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import json
from datetime import datetime

from small_dataset_config import SmallDatasetConfig
from lightweight_model import SmallDatasetXrayDRRModel
from lightweight_losses import SmallDatasetLoss
from drr_dataset_loading import DRRDataset
from util import dice_coefficient, jaccard_index


def train_lightweight_model():
    """Train the lightweight model with small dataset optimizations."""
    config = SmallDatasetConfig()
    config.create_directories()
    
    print(f"Training configuration: {config}")
    print(f"Device: {config.DEVICE}")
    
    # Prepare datasets with careful splitting for small datasets
    print("Loading dataset...")
    train_dataset = DRRDataset(
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        training=True,
        augment=config.AUGMENT_DATA,
        normalize=config.NORMALIZE_DATA
    )
    
    val_dataset = DRRDataset(
        data_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        training=False,
        augment=False,  # No augmentation for validation
        normalize=config.NORMALIZE_DATA
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with small batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Important for small datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Initialize lightweight model
    print("Initializing lightweight model...")
    model = SmallDatasetXrayDRRModel(
        alpha=config.ALPHA,
        freeze_early_layers=True  # Heavy regularization
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss function
    criterion = SmallDatasetLoss(
        pos_weight=config.POS_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        bce_weight=config.BCE_WEIGHT
    ).to(config.DEVICE)
    
    # Initialize optimizer with moderate learning rate
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        verbose=True
    )
    
    # Training tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'learning_rates': []
    }
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        for batch_idx, (xray, drr, mask) in enumerate(train_pbar):
            xray = xray.to(config.DEVICE)
            drr = drr.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(xray, drr)
            loss = criterion(outputs, mask)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_mask = torch.sigmoid(outputs) > 0.5
                dice = dice_coefficient(pred_mask, mask)
                
                train_loss += loss.item()
                train_dice += dice.item()
                num_train_batches += 1
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / num_train_batches
        avg_train_dice = train_dice / num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
            for xray, drr, mask in val_pbar:
                xray = xray.to(config.DEVICE)
                drr = drr.to(config.DEVICE)
                mask = mask.to(config.DEVICE)
                
                outputs = model(xray, drr)
                loss = criterion(outputs, mask)
                
                pred_mask = torch.sigmoid(outputs) > 0.5
                dice = dice_coefficient(pred_mask, mask)
                
                val_loss += loss.item()
                val_dice += dice.item()
                num_val_batches += 1
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}'
                })
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / num_val_batches
        avg_val_dice = val_dice / num_val_batches
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['learning_rates'].append(current_lr)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'config': config,
                'history': history
            }, os.path.join(config.SAVE_DIR, 'best_lightweight_model.pth'))
            
            print(f'  → New best model saved! Dice: {best_dice:.4f}')
        else:
            patience_counter += 1
            print(f'  → No improvement ({patience_counter}/{config.PATIENCE})')
        
        # Early stopping for small datasets
        if patience_counter >= config.PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        print('-' * 50)
    
    print(f'\nTraining completed!')
    print(f'Best Dice score: {best_dice:.4f} at epoch {best_epoch}')
    
    # Save final model and history
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_dice': avg_val_dice,
        'config': config,
        'history': history
    }, os.path.join(config.SAVE_DIR, 'final_lightweight_model.pth'))
    
    # Save training history
    with open(os.path.join(config.LOG_DIR, 'lightweight_training_history.json'), 'w') as f:
        # Convert any tensors to lists for JSON serialization
        json_history = {}
        for key, values in history.items():
            if isinstance(values[0], torch.Tensor):
                json_history[key] = [float(v.cpu()) for v in values]
            else:
                json_history[key] = [float(v) for v in values]
        
        json.dump({
            'history': json_history,
            'best_dice': float(best_dice),
            'best_epoch': int(best_epoch),
            'total_epochs': epoch + 1,
            'config': str(config)
        }, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, config.LOG_DIR)
    
    return model, history, best_dice


def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Dice curves
    ax2.plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Training and Validation Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Validation Dice zoomed
    ax4.plot(epochs, history['val_dice'], 'r-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Dice Score')
    ax4.set_title('Validation Dice Score (Detailed)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lightweight_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir}/lightweight_training_curves.png")


if __name__ == "__main__":
    model, history, best_dice = train_lightweight_model()
    print(f"Final best Dice score: {best_dice:.4f}")
