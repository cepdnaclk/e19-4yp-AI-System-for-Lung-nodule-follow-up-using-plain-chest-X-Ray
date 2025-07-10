"""
Improved training script with better architecture and comprehensive visualization.
"""

import torchxrayvision as xrv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import numpy as np

from drr_dataset_loading import DRRSegmentationDataset
from custom_model import ImprovedXrayDRRSegmentationModel, XrayDRRSegmentationModel
from util import (dice_loss, show_prediction_vs_groundtruth, hybrid_loss, 
                  find_optimal_threshold, calculate_metrics, focal_loss)
from visualization import ModelVisualizer, visualize_dataset_statistics
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    """Enhanced trainer with better loss functions and visualization."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        
        # Create directories
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
    def setup_data(self):
        """Setup datasets and dataloaders."""
        logger.info("Setting up datasets...")
        
        # Create dataset
        dataset = DRRSegmentationDataset(
            root_dir=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            augment=self.config.AUGMENT_DATA,
            normalize=self.config.NORMALIZE_DATA
        )
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Visualize dataset statistics
        visualize_dataset_statistics(dataset, 
                                   save_path=os.path.join(self.config.RESULTS_DIR, 'dataset_stats.png'))
        
        # Split dataset
        train_size = int(self.config.TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return dataset
        
    def setup_model(self):
        """Setup model and optimizer."""
        logger.info("Setting up model...")
        
        # Load pretrained model
        try:
            pretrained_model = xrv.models.ResNet(weights=self.config.PRETRAINED_WEIGHTS)
            logger.info(f"Loaded pretrained model: {self.config.PRETRAINED_WEIGHTS}")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise
        
        # Initialize improved model
        self.model = ImprovedXrayDRRSegmentationModel(
            pretrained_model, 
            alpha=self.config.ALPHA,
            use_channel_attention=True
        ).to(self.device)
        
        # Setup optimizer
        if self.config.OPTIMIZER.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
        
        # Setup scheduler
        if self.config.SCHEDULER == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', 
                patience=self.config.SCHEDULER_PATIENCE,
                factor=self.config.SCHEDULER_FACTOR
            )
        elif self.config.SCHEDULER == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.NUM_EPOCHS
            )
        
        # Setup visualizer
        self.visualizer = ModelVisualizer(self.model, self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
        
    def compute_loss(self, pred, mask, attention_map=None):
        """Compute improved loss function."""
        # Main segmentation loss
        if self.config.USE_FOCAL_LOSS:
            seg_loss = self.config.FOCAL_WEIGHT * focal_loss(pred, mask, alpha=1, gamma=2)
            seg_loss += self.config.DICE_WEIGHT * dice_loss(pred, mask)
        else:
            seg_loss = hybrid_loss(pred, mask, pos_weight=15.0)
        
        # Attention loss (if attention map is provided)
        if attention_map is not None:
            attn_resized = F.interpolate(attention_map, size=mask.shape[2:], 
                                       mode='bilinear', align_corners=False)
            attn_loss = F.binary_cross_entropy(attn_resized, mask)
            total_loss = seg_loss + self.config.LAMBDA_ATTN * attn_loss
            return total_loss, seg_loss, attn_loss
        
        return seg_loss, seg_loss, 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        train_dice = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                xray = batch["xray"].to(self.device)
                drr = batch["drr"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                result = self.model(xray, drr)
                
                if isinstance(result, dict):
                    pred_mask = result['segmentation']
                    attention_map = result['attention']
                else:
                    pred_mask = result
                    attention_map = None
                
                # Compute loss
                total_loss, seg_loss, attn_loss = self.compute_loss(pred_mask, mask, attention_map)
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    pred_binary = (pred_mask > 0.5).float()
                    batch_dice = calculate_metrics(pred_binary, mask)['dice']
                
                train_loss += total_loss.item()
                train_dice += batch_dice
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}, Batch {batch_idx}/{len(self.train_loader)}, "
                              f"Loss: {total_loss.item():.4f}, Seg: {seg_loss.item():.4f}, "
                              f"Attn: {attn_loss:.4f}, Dice: {batch_dice:.4f}")
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
        avg_train_dice = train_dice / num_batches if num_batches > 0 else 0.0
        
        return avg_train_loss, avg_train_dice
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    xray = batch["xray"].to(self.device)
                    drr = batch["drr"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    
                    # Forward pass
                    result = self.model(xray, drr)
                    
                    if isinstance(result, dict):
                        pred_mask = result['segmentation']
                        attention_map = result['attention']
                    else:
                        pred_mask = result
                        attention_map = None
                    
                    # Compute loss
                    total_loss, _, _ = self.compute_loss(pred_mask, mask, attention_map)
                    
                    # Calculate metrics
                    pred_binary = (pred_mask > 0.5).float()
                    batch_dice = calculate_metrics(pred_binary, mask)['dice']
                    
                    val_loss += total_loss.item()
                    val_dice += batch_dice
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        avg_val_dice = val_dice / num_batches if num_batches > 0 else 0.0
        
        return avg_val_loss, avg_val_dice
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        early_stopping_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_dice = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_dices.append(train_dice)
            
            # Validate
            val_loss, val_dice = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_dices.append(val_dice)
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                if self.config.SCHEDULER == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                       f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                early_stopping_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(self.config.SAVE_DIR, 'best_model.pth'))
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}, dice: {val_dice:.4f}")
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Periodic visualization
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.visualize_progress(epoch)
    
    def visualize_progress(self, epoch):
        """Visualize training progress."""
        # Get a sample from validation set
        sample_batch = next(iter(self.val_loader))
        xray = sample_batch["xray"][:1].to(self.device)
        drr = sample_batch["drr"][:1].to(self.device)
        mask = sample_batch["mask"][:1].to(self.device)
        
        # Save attention visualization
        save_path = os.path.join(self.config.RESULTS_DIR, f'attention_epoch_{epoch+1}.png')
        self.visualizer.visualize_attention_mechanism(xray, drr, mask, save_path)
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.train_dices, label='Train Dice', color='blue')
        plt.plot(self.val_dices, label='Val Dice', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Training and Validation Dice')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train', alpha=0.7)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, f'training_curves_epoch_{epoch+1}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def post_training_analysis(self):
        """Comprehensive analysis after training."""
        logger.info("Starting post-training analysis...")
        
        # Load best model
        checkpoint = torch.load(os.path.join(self.config.SAVE_DIR, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Comprehensive evaluation
        metrics = self.visualizer.analyze_prediction_quality(
            self.val_loader, 
            num_samples=20, 
            save_path=os.path.join(self.config.RESULTS_DIR, 'quality_analysis.png')
        )
        
        # Threshold analysis
        threshold_metrics, optimal_threshold = self.visualizer.threshold_analysis(
            self.val_loader,
            save_path=os.path.join(self.config.RESULTS_DIR, 'threshold_analysis.png')
        )
        
        # Update checkpoint with optimal threshold
        checkpoint['optimal_threshold'] = optimal_threshold
        checkpoint['threshold_metrics'] = threshold_metrics
        torch.save(checkpoint, os.path.join(self.config.SAVE_DIR, 'best_model.pth'))
        
        # Feature map visualization
        sample_batch = next(iter(self.val_loader))
        xray = sample_batch["xray"][:1].to(self.device)
        drr = sample_batch["drr"][:1].to(self.device)
        
        self.visualizer.visualize_feature_maps(
            xray, drr, 
            save_path=os.path.join(self.config.RESULTS_DIR, 'feature_maps.png')
        )
        
        logger.info("Post-training analysis completed!")
        
        return optimal_threshold, metrics

def main():
    """Main training function."""
    # Configuration
    config = Config()
    
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Configuration: {vars(config)}")
    
    # Initialize trainer
    trainer = ImprovedTrainer(config)
    
    # Setup data and model
    dataset = trainer.setup_data()
    trainer.setup_model()
    
    # Train model
    trainer.train()
    
    # Post-training analysis
    optimal_threshold, metrics = trainer.post_training_analysis()
    
    logger.info("Training completed successfully!")
    logger.info(f"Best model saved with optimal threshold: {optimal_threshold:.4f}")
    
if __name__ == "__main__":
    main()
