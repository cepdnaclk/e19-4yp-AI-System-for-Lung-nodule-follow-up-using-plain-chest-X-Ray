"""
Training script for the improved model with supervised attention.
This should resolve the attention-segmentation alignment issues.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our modules
from improved_model import ImprovedXrayDRRModel
from improved_config import (
    IMPROVED_CONFIG, 
    create_improved_model, 
    create_improved_loss,
    get_improved_optimizer,
    get_improved_scheduler
)
from drr_dataset_loading import DRRDataset
from util import dice_coefficient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    """Trainer for the improved model with attention supervision."""
    
    def __init__(self, config=None):
        self.config = config or IMPROVED_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path('checkpoints_improved')
        self.log_dir = Path('logs_improved')
        self.attention_dir = Path('attention_maps_improved')
        
        for directory in [self.checkpoint_dir, self.log_dir, self.attention_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize model, loss, and optimizer
        self.model = create_improved_model().to(self.device)
        self.criterion = create_improved_loss()
        self.optimizer = get_improved_optimizer(self.model)
        self.scheduler = get_improved_scheduler(self.optimizer)
        
        # Training tracking
        self.best_dice = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []
        self.attention_losses = []
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def create_dataloaders(self):
        """Create training and validation dataloaders."""
        data_root = r"e:\Campus2\fyp-repo\e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray\DRR dataset\LIDC_LDRI"
        
        # Training dataset
        train_dataset = DRRDataset(
            data_root=data_root,
            training=True,
            augment=self.config['AUGMENT_TRAINING'],
            normalize=self.config['NORMALIZE_IMAGES']
        )
        
        # Validation dataset
        val_dataset = DRRDataset(
            data_root=data_root,
            training=False,
            augment=False,
            normalize=self.config['NORMALIZE_IMAGES']
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_attention_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["NUM_EPOCHS"]}')
        
        for batch_idx, (xray, drr, masks) in enumerate(progress_bar):
            xray = xray.to(self.device)
            drr = drr.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with ground truth for attention supervision
            predictions = self.model(xray, drr, ground_truth_mask=masks)
            
            # Compute loss
            loss, loss_info = self.criterion(
                predictions, 
                masks, 
                epoch=epoch, 
                total_epochs=self.config['NUM_EPOCHS']
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                seg_pred = torch.sigmoid(predictions['segmentation'])
                dice = dice_coefficient(seg_pred, masks)
                total_dice += dice.item()
            
            # Update totals
            total_loss += loss.item()
            total_seg_loss += loss_info['segmentation_loss'].item()
            if 'attention_supervision_loss' in loss_info:
                total_attention_loss += loss_info['attention_supervision_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'AttLoss': f'{loss_info.get("attention_supervision_loss", torch.tensor(0)).item():.4f}'
            })
            
            # Log intermediate results
            if batch_idx % self.config['LOG_FREQUENCY'] == 0:
                logger.info(
                    f'Epoch {epoch+1}, Batch {batch_idx}: '
                    f'Loss={loss.item():.4f}, '
                    f'Dice={dice.item():.4f}, '
                    f'AttLoss={loss_info.get("attention_supervision_loss", torch.tensor(0)).item():.4f}'
                )
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_attention_loss = total_attention_loss / num_batches if total_attention_loss > 0 else 0
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_seg_loss, avg_attention_loss, avg_dice
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for xray, drr, masks in val_loader:
                xray = xray.to(self.device)
                drr = drr.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(xray, drr, ground_truth_mask=masks)
                
                # Compute loss
                loss, _ = self.criterion(
                    predictions, 
                    masks, 
                    epoch=epoch, 
                    total_epochs=self.config['NUM_EPOCHS']
                )
                
                # Calculate dice
                seg_pred = torch.sigmoid(predictions['segmentation'])
                dice = dice_coefficient(seg_pred, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        return avg_loss, avg_dice\n    \n    def save_attention_maps(self, val_loader, epoch):\n        \"\"\"Save attention maps for visualization.\"\"\"\n        if epoch % self.config['ATTENTION_SAVE_FREQUENCY'] != 0:\n            return\n            \n        self.model.eval()\n        \n        with torch.no_grad():\n            # Get first batch\n            xray, drr, masks = next(iter(val_loader))\n            xray = xray.to(self.device)\n            drr = drr.to(self.device)\n            masks = masks.to(self.device)\n            \n            predictions = self.model(xray, drr)\n            \n            # Save first sample in batch\n            attention = predictions['attention'][0, 0].cpu().numpy()  # [512, 512]\n            segmentation = torch.sigmoid(predictions['segmentation'][0, 0]).cpu().numpy()  # [512, 512]\n            ground_truth = masks[0, 0].cpu().numpy()  # [512, 512]\n            drr_img = drr[0, 0].cpu().numpy()  # [512, 512]\n            \n            # Create visualization\n            fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n            fig.suptitle(f'Attention Analysis - Epoch {epoch+1}', fontsize=14)\n            \n            # Top row: inputs and outputs\n            axes[0, 0].imshow(drr_img, cmap='gray')\n            axes[0, 0].set_title('DRR Input')\n            axes[0, 0].axis('off')\n            \n            axes[0, 1].imshow(attention, cmap='hot')\n            axes[0, 1].set_title(f'Attention Map\\n(Range: {attention.min():.3f}-{attention.max():.3f})')\n            axes[0, 1].axis('off')\n            \n            axes[0, 2].imshow(segmentation, cmap='gray')\n            axes[0, 2].set_title('Segmentation Output')\n            axes[0, 2].axis('off')\n            \n            # Bottom row: ground truth and overlays\n            axes[1, 0].imshow(ground_truth, cmap='gray')\n            axes[1, 0].set_title('Ground Truth')\n            axes[1, 0].axis('off')\n            \n            # Attention overlay on DRR\n            axes[1, 1].imshow(drr_img, cmap='gray', alpha=0.7)\n            axes[1, 1].imshow(attention, cmap='hot', alpha=0.5)\n            axes[1, 1].set_title('Attention Overlay')\n            axes[1, 1].axis('off')\n            \n            # All overlays\n            axes[1, 2].imshow(ground_truth, cmap='Greens', alpha=0.3)\n            axes[1, 2].imshow(segmentation, cmap='Blues', alpha=0.3)\n            axes[1, 2].imshow(attention, cmap='Reds', alpha=0.3)\n            axes[1, 2].set_title('Combined (G:GT, B:Seg, R:Att)')\n            axes[1, 2].axis('off')\n            \n            plt.tight_layout()\n            \n            # Save\n            save_path = self.attention_dir / f'attention_epoch_{epoch+1}.png'\n            plt.savefig(save_path, dpi=150, bbox_inches='tight')\n            plt.close()\n            \n            logger.info(f\"Attention maps saved to {save_path}\")\n    \n    def save_checkpoint(self, epoch, is_best=False):\n        \"\"\"Save model checkpoint.\"\"\"\n        checkpoint = {\n            'epoch': epoch,\n            'model_state_dict': self.model.state_dict(),\n            'optimizer_state_dict': self.optimizer.state_dict(),\n            'scheduler_state_dict': self.scheduler.state_dict(),\n            'best_dice': self.best_dice,\n            'train_losses': self.train_losses,\n            'val_losses': self.val_losses,\n            'val_dices': self.val_dices,\n            'config': self.config\n        }\n        \n        # Save regular checkpoint\n        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'\n        torch.save(checkpoint, checkpoint_path)\n        \n        # Save best model\n        if is_best:\n            best_path = self.checkpoint_dir / 'best_model_improved.pth'\n            torch.save(checkpoint, best_path)\n            logger.info(f\"New best model saved with Dice: {self.best_dice:.4f}\")\n    \n    def train(self):\n        \"\"\"Main training loop.\"\"\"\n        logger.info(\"Starting improved model training...\")\n        \n        # Create dataloaders\n        train_loader, val_loader = self.create_dataloaders()\n        \n        # Training loop\n        patience_counter = 0\n        \n        for epoch in range(self.config['NUM_EPOCHS']):\n            logger.info(f\"\\nEpoch {epoch+1}/{self.config['NUM_EPOCHS']}\")\n            \n            # Train\n            train_loss, train_seg_loss, train_att_loss, train_dice = self.train_epoch(train_loader, epoch)\n            \n            # Validate\n            if epoch % self.config['VALIDATION_FREQUENCY'] == 0:\n                val_loss, val_dice = self.validate(val_loader, epoch)\n                \n                # Update learning rate\n                self.scheduler.step(val_loss)\n                \n                # Track metrics\n                self.train_losses.append(train_loss)\n                self.val_losses.append(val_loss)\n                self.val_dices.append(val_dice)\n                self.attention_losses.append(train_att_loss)\n                \n                # Check for best model\n                is_best = val_dice > self.best_dice\n                if is_best:\n                    self.best_dice = val_dice\n                    self.best_epoch = epoch\n                    patience_counter = 0\n                else:\n                    patience_counter += 1\n                \n                # Save checkpoint\n                if self.config['SAVE_BEST_MODEL']:\n                    self.save_checkpoint(epoch, is_best)\n                \n                # Save attention maps\n                if self.config['SAVE_ATTENTION_MAPS']:\n                    self.save_attention_maps(val_loader, epoch)\n                \n                # Log results\n                logger.info(\n                    f\"Epoch {epoch+1}: \"\n                    f\"Train Loss: {train_loss:.4f}, \"\n                    f\"Train Dice: {train_dice:.4f}, \"\n                    f\"Val Loss: {val_loss:.4f}, \"\n                    f\"Val Dice: {val_dice:.4f}, \"\n                    f\"Att Loss: {train_att_loss:.4f}\"\n                )\n                \n                # Early stopping\n                if patience_counter >= self.config['EARLY_STOPPING_PATIENCE']:\n                    logger.info(f\"Early stopping at epoch {epoch+1}\")\n                    break\n        \n        logger.info(f\"Training completed. Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch+1}\")\n        \n        # Save final training curves\n        self.save_training_curves()\n    \n    def save_training_curves(self):\n        \"\"\"Save training curves.\"\"\"\n        fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n        \n        # Loss curves\n        axes[0, 0].plot(self.train_losses, label='Train Loss')\n        axes[0, 0].plot(self.val_losses, label='Val Loss')\n        axes[0, 0].set_title('Loss Curves')\n        axes[0, 0].set_xlabel('Epoch')\n        axes[0, 0].set_ylabel('Loss')\n        axes[0, 0].legend()\n        axes[0, 0].grid(True)\n        \n        # Dice curve\n        axes[0, 1].plot(self.val_dices, label='Val Dice', color='green')\n        axes[0, 1].set_title('Dice Coefficient')\n        axes[0, 1].set_xlabel('Epoch')\n        axes[0, 1].set_ylabel('Dice')\n        axes[0, 1].legend()\n        axes[0, 1].grid(True)\n        \n        # Attention loss\n        if self.attention_losses:\n            axes[1, 0].plot(self.attention_losses, label='Attention Loss', color='red')\n            axes[1, 0].set_title('Attention Supervision Loss')\n            axes[1, 0].set_xlabel('Epoch')\n            axes[1, 0].set_ylabel('Attention Loss')\n            axes[1, 0].legend()\n            axes[1, 0].grid(True)\n        \n        # Learning rate (if available)\n        current_lr = self.optimizer.param_groups[0]['lr']\n        axes[1, 1].text(0.5, 0.5, f'Final LR: {current_lr:.2e}\\nBest Dice: {self.best_dice:.4f}\\nBest Epoch: {self.best_epoch+1}', \n                        transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12,\n                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n        axes[1, 1].set_title('Training Summary')\n        axes[1, 1].axis('off')\n        \n        plt.tight_layout()\n        \n        curves_path = self.log_dir / 'training_curves_improved.png'\n        plt.savefig(curves_path, dpi=150, bbox_inches='tight')\n        plt.close()\n        \n        logger.info(f\"Training curves saved to {curves_path}\")\n\ndef main():\n    \"\"\"Main training function.\"\"\"\n    # Create trainer\n    trainer = ImprovedTrainer()\n    \n    # Start training\n    trainer.train()\n    \n    print(\"\\n\" + \"=\"*60)\n    print(\"IMPROVED TRAINING COMPLETED!\")\n    print(\"=\"*60)\n    print(f\"Best Dice Score: {trainer.best_dice:.4f}\")\n    print(f\"Best Epoch: {trainer.best_epoch+1}\")\n    print(f\"Model saved to: checkpoints_improved/best_model_improved.pth\")\n    print(f\"Attention maps saved to: attention_maps_improved/\")\n    print(f\"Training curves saved to: logs_improved/training_curves_improved.png\")\n    print(\"\\nTo visualize results, run:\")\n    print(\"python visualize_model_internals.py --model checkpoints_improved/best_model_improved.pth --sample 0 --from-training\")\n\nif __name__ == \"__main__\":\n    main()"}
