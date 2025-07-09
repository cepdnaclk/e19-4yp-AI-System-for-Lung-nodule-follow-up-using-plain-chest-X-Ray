"""
Simplified loss functions focused on medical segmentation accuracy.
Removes unnecessary complexity while maintaining effectiveness.
"""

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """Dice loss with numerical stability."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def combined_loss(pred, target, dice_weight=0.7, focal_weight=0.3):
    """Simplified combined loss focusing on dice and focal."""
    d_loss = dice_loss(pred, target)
    f_loss = focal_loss(pred, target)
    return dice_weight * d_loss + focal_weight * f_loss

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate comprehensive metrics for evaluation."""
    pred_binary = (pred > threshold).float()
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Basic metrics
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    # Calculated metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Dice coefficient
    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    
    # IoU
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def find_optimal_threshold(pred, target, thresholds=None):
    """Find optimal threshold based on dice score."""
    if thresholds is None:
        thresholds = torch.linspace(0.1, 0.9, 17)
    
    best_threshold = 0.5
    best_dice = 0.0
    
    for threshold in thresholds:
        metrics = calculate_metrics(pred, target, threshold)
        dice = metrics['dice']
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold.item()
    
    return best_threshold, best_dice
