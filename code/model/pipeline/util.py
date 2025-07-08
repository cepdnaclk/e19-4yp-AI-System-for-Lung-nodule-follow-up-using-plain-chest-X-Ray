import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

def jaccard_score_manual(y_true, y_pred):
    """Manual implementation of Jaccard score (IoU) to avoid sklearn dependency."""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def show_prediction_vs_groundtruth(xrays, preds, masks, num=4, save_path=None):
    """Enhanced visualization with better layout and metrics display."""
    plt.figure(figsize=(20, 6))
    
    for i in range(min(num, len(xrays))):
        # Convert tensors to numpy for visualization
        xray_np = xrays[i][0].cpu().numpy()
        pred_np = preds[i][0].cpu().numpy()
        mask_np = masks[i][0].cpu().numpy()
        
        # Calculate metrics for this sample
        pred_binary = (pred_np > 0.5).astype(np.float32)
        dice = dice_coefficient(torch.tensor(pred_binary), torch.tensor(mask_np))
        iou = jaccard_score_manual(mask_np.flatten(), pred_binary.flatten())
        
        # X-ray
        plt.subplot(3, num, i+1)
        plt.imshow(xray_np, cmap='gray')
        plt.title(f"X-ray {i+1}")
        plt.axis('off')

        # Predicted mask
        plt.subplot(3, num, num+i+1)
        plt.imshow(pred_np, cmap='hot', vmin=0, vmax=1)
        plt.title(f"Predicted\nDice: {dice:.3f}")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(3, num, 2*num+i+1)
        plt.imshow(mask_np, cmap='hot', vmin=0, vmax=1)
        plt.title(f"Ground Truth\nIoU: {iou:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

def dice_loss(pred, target, eps=1e-7):
    """Improved dice loss with numerical stability."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice

def dice_coefficient(pred, target, eps=1e-7):
    """Calculate dice coefficient metric."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

def focal_loss(pred, target, alpha=1, gamma=2, eps=1e-7):
    """Focal loss for handling class imbalance."""
    ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def combined_loss(pred, target, dice_weight=0.5, focal_weight=0.5):
    """Combined loss function using dice loss and focal loss."""
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    return dice_weight * d_loss + focal_weight * f_loss

def weighted_bce_loss(pred, target, pos_weight=10.0):
    """Weighted Binary Cross Entropy to handle class imbalance."""
    # Calculate positive weight based on class distribution
    pos_pixels = target.sum()
    neg_pixels = target.numel() - pos_pixels
    
    if pos_pixels > 0:
        calculated_pos_weight = neg_pixels / pos_pixels
        pos_weight = min(pos_weight, calculated_pos_weight)
    
    return F.binary_cross_entropy(pred, target, 
                                 weight=target * pos_weight + (1 - target))

def hybrid_loss(pred, target, dice_weight=0.3, focal_weight=0.3, bce_weight=0.4, pos_weight=10.0):
    """Hybrid loss combining dice, focal, and weighted BCE."""
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    w_bce_loss = weighted_bce_loss(pred, target, pos_weight)
    
    return dice_weight * d_loss + focal_weight * f_loss + bce_weight * w_bce_loss

def find_optimal_threshold(pred, target, thresholds=None):
    """Find optimal threshold for binary segmentation."""
    if thresholds is None:
        thresholds = torch.linspace(0.1, 0.9, 17)
    
    best_threshold = 0.5
    best_dice = 0.0
    
    for threshold in thresholds:
        pred_binary = (pred > threshold).float()
        dice = dice_coefficient(pred_binary, target)
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold.item()
    
    return best_threshold, best_dice

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate comprehensive metrics for segmentation."""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection) / (union + 1e-7)
    iou = jaccard_score_manual(target_flat, pred_flat)
    
    # Precision, Recall, F1
    tp = intersection
    fp = pred_flat.sum() - intersection
    fn = target_flat.sum() - intersection
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 4))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train', color='blue')
    plt.plot(epochs, val_losses, label='Validation', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()