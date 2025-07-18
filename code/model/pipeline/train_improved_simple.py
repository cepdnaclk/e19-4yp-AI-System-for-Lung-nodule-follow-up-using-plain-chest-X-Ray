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
from training_visualization import TrainingVisualizer, EvaluationVisualizer

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
        
        # Initialize visualization
        self.visualizer = TrainingVisualizer(save_dir='training_visualizations')
        
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
        # Use relative path that works across platforms
        data_root = os.path.join("..", "..", "..", "DRR dataset", "LIDC_LDRI")
        
        # If relative path doesn't exist, try absolute path detection
        if not os.path.exists(data_root):
            # Try to find the dataset in the project structure
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            data_root = os.path.join(project_root, "DRR dataset", "LIDC_LDRI")
            
            if not os.path.exists(data_root):
                # Last resort: look for any LIDC_LDRI directory
                for root, dirs, files in os.walk(os.path.dirname(project_root)):
                    if "LIDC_LDRI" in dirs:
                        data_root = os.path.join(root, "LIDC_LDRI")
                        break
                        
        logger.info(f"Using dataset path: {data_root}")
        
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
                # Handle both tensor and float returns from dice_coefficient
                dice_value = dice.item() if hasattr(dice, 'item') else dice
                total_dice += dice_value
            
            # Update totals
            total_loss += loss.item()
            total_seg_loss += loss_info['segmentation_loss'].item()
            if 'attention_supervision_loss' in loss_info:
                total_attention_loss += loss_info['attention_supervision_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice_value:.4f}',
                'AttLoss': f'{loss_info.get("attention_supervision_loss", torch.tensor(0)).item():.4f}'
            })
            
            # Log intermediate results
            if batch_idx % self.config['LOG_FREQUENCY'] == 0:
                logger.info(
                    f'Epoch {epoch+1}, Batch {batch_idx}: '
                    f'Loss={loss.item():.4f}, '
                    f'Dice={dice_value:.4f}, '
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
        total_seg_loss = 0.0
        total_attention_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for xray, drr, masks in val_loader:
                xray = xray.to(self.device)
                drr = drr.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(xray, drr, ground_truth_mask=masks)
                
                # Compute loss
                loss, loss_info = self.criterion(
                    predictions, 
                    masks, 
                    epoch=epoch, 
                    total_epochs=self.config['NUM_EPOCHS']
                )
                
                # Calculate dice
                seg_pred = torch.sigmoid(predictions['segmentation'])
                dice = dice_coefficient(seg_pred, masks)
                # Handle both tensor and float returns from dice_coefficient
                dice_value = dice.item() if hasattr(dice, 'item') else dice
                
                total_loss += loss.item()
                total_dice += dice_value
                total_seg_loss += loss_info['segmentation_loss'].item()
                if 'attention_supervision_loss' in loss_info:
                    total_attention_loss += loss_info['attention_supervision_loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_attention_loss = total_attention_loss / num_batches if total_attention_loss > 0 else 0
        
        return avg_loss, avg_seg_loss, avg_attention_loss, avg_dice
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dices': self.val_dices,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model_improved.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with Dice: {self.best_dice:.4f}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting improved model training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config['NUM_EPOCHS']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['NUM_EPOCHS']}")
            
            # Train
            train_loss, train_seg_loss, train_att_loss, train_dice = self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % self.config['VALIDATION_FREQUENCY'] == 0:
                val_loss, val_seg_loss, val_att_loss, val_dice = self.validate(val_loader, epoch)
                
                # Update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                
                # Update visualizer with metrics
                train_metrics = (train_loss, train_seg_loss, train_att_loss, train_dice)
                val_metrics = (val_loss, val_seg_loss, val_att_loss, val_dice)
                self.visualizer.update_metrics(epoch, train_metrics, val_metrics, current_lr)
                
                # Track metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_dices.append(val_dice)
                self.attention_losses.append(train_att_loss)
                
                # Check for best model
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice
                    self.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                if self.config['SAVE_BEST_MODEL']:
                    self.save_checkpoint(epoch, is_best)
                
                # Log results
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Dice: {train_dice:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Dice: {val_dice:.4f}, "
                    f"Att Loss: {train_att_loss:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
                
                # Early stopping
                if patience_counter >= self.config['EARLY_STOPPING_PATIENCE']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # Just log training metrics on non-validation epochs
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Dice: {train_dice:.4f}, "
                    f"Att Loss: {train_att_loss:.4f}"
                )
        
        logger.info(f"Training completed. Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch+1}")
        
        # Generate comprehensive visualizations
        logger.info("Generating training visualizations...")
        try:
            self.visualizer.plot_training_curves()
            self.visualizer.plot_metrics_distribution()
            self.visualizer.save_metrics_csv()
            training_report = self.visualizer.generate_training_report()
            logger.info("✓ Training visualizations completed successfully!")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
        
        return training_report

def main():
    """Main training function."""
    # Create trainer
    trainer = ImprovedTrainer()
    
    # Start training
    training_report = trainer.train()
    
    print("\n" + "="*60)
    print("IMPROVED TRAINING COMPLETED!")
    print("="*60)
    print(f"Best Dice Score: {trainer.best_dice:.4f}")
    print(f"Best Epoch: {trainer.best_epoch+1}")
    print(f"Model saved to: checkpoints_improved/best_model_improved.pth")
    print(f"Training visualizations saved to: training_visualizations/")
    print("="*60)
    
    # Run evaluation on the best model
    print("\nRunning model evaluation...")
    try:
        from training_visualization import EvaluationVisualizer, create_final_visualization_report
        
        # Load best model for evaluation
        best_model_path = 'checkpoints_improved/best_model_improved.pth'
        if os.path.exists(best_model_path):
            # Create evaluation visualizer
            eval_visualizer = EvaluationVisualizer(save_dir='evaluation_visualizations')
            
            # Load validation dataset for evaluation
            from drr_dataset_loading import DRRDataset
            # Use same data path detection logic as trainer
            data_root = "../../DRR dataset/LIDC_LDRI"
            if not os.path.exists(data_root):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                data_root = os.path.join(project_root, "DRR dataset", "LIDC_LDRI")
                
                if not os.path.exists(data_root):
                    for root, dirs, files in os.walk(os.path.dirname(project_root)):
                        if "LIDC_LDRI" in dirs:
                            data_root = os.path.join(root, "LIDC_LDRI")
                            break
            
            val_dataset = DRRDataset(
                data_root=data_root,
                training=False,
                augment=False,
                normalize=IMPROVED_CONFIG['NORMALIZE_IMAGES']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=IMPROVED_CONFIG['BATCH_SIZE'],
                shuffle=False,
                num_workers=2
            )
            
            # Load model
            model = create_improved_model()
            checkpoint = torch.load(best_model_path, map_location=trainer.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(trainer.device)
            
            # Run evaluation
            eval_report = eval_visualizer.evaluate_model_detailed(model, val_loader, trainer.device)
            
            # Create final comprehensive report
            create_final_visualization_report()
            
            print("✓ Model evaluation completed!")
            print(f"Evaluation visualizations saved to: evaluation_visualizations/")
            print(f"Final comprehensive report saved to: final_report/")
            
        else:
            print(f"⚠️ Best model not found at {best_model_path}")
            
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        print("Training completed successfully, but evaluation failed.")
    print("To test the improved model, run:")
    print("python test_improved_model.py")

if __name__ == "__main__":
    main()
