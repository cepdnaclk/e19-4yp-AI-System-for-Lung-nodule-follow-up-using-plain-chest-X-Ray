"""
Improved training script with advanced loss functions and better optimization strategies.
Designed specifically for lung nodule segmentation with extreme class imbalance.
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
from custom_model import ImprovedXrayDRRSegmentationModel
from improved_losses import get_loss_function
from config_ import Config
from util import dice_coefficient, show_prediction_vs_groundtruth, calculate_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    """Enhanced trainer with advanced optimization strategies."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.best_val_loss = float('inf')
        self.best_dice = 0.0
        self.patience_counter = 0
        
        # Create directories
        config.create_directories()
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize model
        self._setup_model()
        
        # Initialize loss function
        self._setup_loss_function()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        
    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        try:
            dataset = DRRSegmentationDataset(
                root_dir=self.config.DATA_ROOT,
                image_size=self.config.IMAGE_SIZE
            )
            logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
            
            # Split dataset into train/val
            train_size = int(self.config.TRAIN_SPLIT * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY
            )
            
            self.val_loader = DataLoader(
                self.val_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY
            )
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _setup_model(self):
        """Initialize the model."""
        try:
            pretrained_model = xrv.models.ResNet(weights=self.config.PRETRAINED_WEIGHTS)
            self.model = ImprovedXrayDRRSegmentationModel(
                pretrained_model, 
                alpha=self.config.ALPHA,
                use_channel_attention=self.config.USE_CHANNEL_ATTENTION
            )
            self.model = self.model.to(self.device)
            logger.info("Model initialized successfully")
            
            # Log model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _setup_loss_function(self):
        """Setup the improved loss function."""
        self.criterion = get_loss_function(
            loss_type=self.config.LOSS_TYPE,
            tversky_weight=self.config.TVERSKY_WEIGHT,
            focal_weight=self.config.FOCAL_WEIGHT,
            bce_weight=self.config.BCE_WEIGHT,
            tversky_alpha=self.config.TVERSKY_ALPHA,
            tversky_beta=self.config.TVERSKY_BETA,
            pos_weight_multiplier=self.config.POS_WEIGHT_MULTIPLIER
        )
        logger.info(f"Loss function initialized: {self.config.LOSS_TYPE}")
    
    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates."""
        # Separate parameters for different learning rates
        backbone_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'feature_extractor' in name:
                    backbone_params.append(param)
                else:
                    new_params.append(param)
        
        # Differential learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.config.BACKBONE_LEARNING_RATE},
            {'params': new_params, 'lr': self.config.LEARNING_RATE}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
            verbose=True
        )
        
        logger.info(f"Optimizer setup with backbone LR: {self.config.BACKBONE_LEARNING_RATE}, "
                   f"new layers LR: {self.config.LEARNING_RATE}")
    
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        train_batches = 0
        loss_components = {'tversky': 0, 'focal': 0, 'bce': 0}
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                xray = batch["xray"].to(self.device)
                drr = batch["drr"].to(self.device)
                mask = batch["mask"].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                result = self.model(xray, drr)
                seg_out = result['segmentation']
                attn_map = result.get('attention', None)

                # Main segmentation loss
                if self.config.LOSS_TYPE == 'combined':
                    loss_seg, loss_dict = self.criterion(seg_out, mask)
                    # Accumulate loss components for monitoring
                    for key in loss_components:
                        if key in loss_dict:
                            loss_components[key] += loss_dict[key]
                else:
                    loss_seg = self.criterion(seg_out, mask)
                
                # Light attention supervision
                loss_total = loss_seg
                if attn_map is not None:
                    attn_resized = F.interpolate(
                        attn_map, size=mask.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                    loss_attn = F.binary_cross_entropy(attn_resized, mask)
                    loss_total = loss_seg + self.config.LAMBDA_ATTN * loss_attn

                # Backward and optimize with gradient clipping
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
                self.optimizer.step()

                train_loss += loss_total.item()
                train_batches += 1
                
                if batch_idx % self.config.LOG_INTERVAL == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}, "
                              f"Batch {batch_idx}/{len(self.train_loader)}, "
                              f"Loss: {loss_total.item():.6f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Log average loss components
        if loss_components['tversky'] > 0:
            logger.info(f"Epoch {epoch+1} Loss Components - "
                       f"Tversky: {loss_components['tversky']/train_batches:.6f}, "
                       f"Focal: {loss_components['focal']/train_batches:.6f}, "
                       f"BCE: {loss_components['bce']/train_batches:.6f}")
        
        return avg_train_loss
    
    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_batches = 0
        dice_scores = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    xray = batch["xray"].to(self.device)
                    drr = batch["drr"].to(self.device)
                    mask = batch["mask"].to(self.device)

                    result = self.model(xray, drr)
                    seg_out = result['segmentation']
                    attn_map = result.get('attention', None)
                    
                    # Calculate loss
                    if self.config.LOSS_TYPE == 'combined':
                        loss_seg, _ = self.criterion(seg_out, mask)
                    else:
                        loss_seg = self.criterion(seg_out, mask)
                    
                    loss_total = loss_seg
                    if attn_map is not None:
                        attn_resized = F.interpolate(
                            attn_map, size=mask.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                        loss_attn = F.binary_cross_entropy(attn_resized, mask)
                        loss_total = loss_seg + self.config.LAMBDA_ATTN * loss_attn

                    val_loss += loss_total.item()
                    val_batches += 1
                    
                    # Calculate Dice scores
                    for i in range(seg_out.shape[0]):
                        pred_binary = (seg_out[i] > 0.5).float()
                        dice = dice_coefficient(pred_binary, mask[i])
                        dice_scores.append(dice)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        
        return avg_val_loss, avg_dice
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_dice': self.best_dice,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.SAVE_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with Dice: {self.best_dice:.4f}")
    
    def _visualize_predictions(self, epoch):
        """Visualize predictions during training."""
        if epoch % self.config.VISUALIZATION_FREQUENCY != 0:
            return
        
        self.model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.val_loader))
            xray = batch["xray"].to(self.device)
            drr = batch["drr"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            result = self.model(xray, drr)
            seg_out = result['segmentation']
            attn_map = result.get('attention', None)
            
            # Save visualization
            save_path = os.path.join(self.config.RESULTS_DIR, f'predictions_epoch_{epoch+1}.png')
            show_prediction_vs_groundtruth(xray, seg_out, mask, num=4, save_path=save_path)
            
            # Save attention maps if available
            if attn_map is not None:
                attn_save_path = os.path.join(self.config.RESULTS_DIR, f'attention_epoch_{epoch+1}.png')
                self._save_attention_maps(drr, attn_map, attn_save_path)
    
    def _save_attention_maps(self, drr, attn_map, save_path):
        """Save attention map visualizations."""
        plt.figure(figsize=(16, 4))
        
        for i in range(min(4, drr.shape[0])):
            # DRR image
            plt.subplot(2, 4, i+1)
            plt.imshow(drr[i][0].cpu().numpy(), cmap='gray')
            plt.title(f"DRR {i+1}")
            plt.axis('off')
            
            # Attention map
            plt.subplot(2, 4, i+5)
            plt.imshow(attn_map[i][0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
            plt.title(f"Attention {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training on {len(self.train_dataset)} samples, "
                   f"Validating on {len(self.val_dataset)} samples")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train epoch
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate epoch
            val_loss, val_dice = self._validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_dice_scores.append(val_dice)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"Val Dice: {val_dice:.4f}")
            
            # Check for best model
            is_best = False
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.best_val_loss = val_loss
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            
            # Visualize predictions
            self._visualize_predictions(epoch)
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training completed! Best Dice: {self.best_dice:.4f}")
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Dice scores
        plt.subplot(1, 3, 2)
        plt.plot(self.val_dice_scores, label='Val Dice', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 3, 3)
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(lrs, label='Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(self.config.RESULTS_DIR, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize trainer
    trainer = ImprovedTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
