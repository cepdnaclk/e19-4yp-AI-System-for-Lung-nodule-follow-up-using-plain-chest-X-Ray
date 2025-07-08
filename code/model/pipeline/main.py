import torchxrayvision as xrv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

from drr_dataset_loading import DRRSegmentationDataset
from custom_model import XrayDRRSegmentationModel
from util import dice_loss, show_prediction_vs_groundtruth

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def main():
    # Configuration
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (512, 512),
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'lambda_attn': 0.5,
        'alpha': 0.5,
        'save_dir': './checkpoints',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    logger.info(f"Using device: {config['device']}")
    
    # Dataset and DataLoader
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config['data_root'],
            image_size=config['image_size']
        )
        logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Split dataset into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Load pretrained model
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        logger.info("Pretrained model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        return

    # Initialize model
    model = XrayDRRSegmentationModel(pretrained_model, alpha=config['alpha'])
    model = model.to(config['device'])
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
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
                seg_out = model(xray, drr)

                # Get attention map for attention loss
                with torch.no_grad():
                    attn_map = model.attention_net(drr)
                    # Resize attention map to match mask size
                    attn_map = F.interpolate(attn_map, size=mask.shape[2:], mode='bilinear', align_corners=False)

                # Losses
                loss_task = dice_loss(seg_out, mask)
                loss_attn = F.binary_cross_entropy(attn_map, mask)
                loss_total = loss_task + config['lambda_attn'] * loss_attn

                # Backward and optimize
                loss_total.backward()
                optimizer.step()

                train_loss += loss_total.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx}, "
                              f"Loss: {loss_total.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    xray = batch["xray"].to(config['device'])
                    drr = batch["drr"].to(config['device'])
                    mask = batch["mask"].to(config['device'])

                    seg_out = model(xray, drr)
                    attn_map = model.attention_net(drr)
                    attn_map = F.interpolate(attn_map, size=mask.shape[2:], mode='bilinear', align_corners=False)

                    loss_task = dice_loss(seg_out, mask)
                    loss_attn = F.binary_cross_entropy(attn_map, mask)
                    loss_total = loss_task + config['lambda_attn'] * loss_attn

                    val_loss += loss_total.item()
                    val_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                   f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model.pth'))
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.close()
    
    # Test visualization
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(val_loader))
        xray = test_batch["xray"].to(config['device'])
        drr = test_batch["drr"].to(config['device'])
        mask = test_batch["mask"].to(config['device'])
        
        pred_mask = model(xray, drr)
        show_prediction_vs_groundtruth(xray, pred_mask, mask, num=min(4, len(xray)))
        plt.savefig(os.path.join(config['save_dir'], 'predictions.png'))
        plt.close()

if __name__ == "__main__":
    main()