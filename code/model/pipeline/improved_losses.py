"""
Improved loss functions for medical image segmentation with extreme class imbalance.
Specifically designed for lung nodule segmentation where positive pixels are very rare.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Tversky Loss for better handling of precision-recall trade-off.
    Especially effective for small object segmentation like lung nodules.
    
    Args:
        alpha: Weight for False Positives (typically 0.3-0.7)
        beta: Weight for False Negatives (typically 0.3-0.7)
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss combines the benefits of Focal Loss and Tversky Loss.
    Excellent for extremely imbalanced medical segmentation tasks.
    """
    
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky

class DiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal Loss optimized for medical segmentation.
    Dice helps with overlap, Focal helps with hard examples.
    """
    
    def __init__(self, dice_weight=0.4, focal_weight=0.6, alpha=0.25, gamma=2.0, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
    def dice_loss(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice
    
    def focal_loss(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal

class WeightedBCELoss(nn.Module):
    """
    Heavily weighted BCE loss for extreme class imbalance.
    Uses dynamic weighting based on actual class distribution.
    """
    
    def __init__(self, pos_weight_multiplier=100.0, max_pos_weight=1000.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight_multiplier = pos_weight_multiplier
        self.max_pos_weight = max_pos_weight
        
    def forward(self, pred, target):
        # Calculate dynamic positive weight based on class distribution
        pos_pixels = target.sum()
        neg_pixels = target.numel() - pos_pixels
        
        if pos_pixels > 0:
            pos_weight = (neg_pixels / pos_pixels) * self.pos_weight_multiplier
            pos_weight = min(pos_weight, self.max_pos_weight)
        else:
            pos_weight = self.max_pos_weight
        
        # Create weight tensor
        weight = torch.ones_like(target)
        weight[target == 1] = pos_weight
        
        loss = F.binary_cross_entropy(pred, target, weight=weight)
        return loss

class CombinedMedicalLoss(nn.Module):
    """
    Ultimate loss function combining multiple loss types optimized for lung nodule segmentation.
    This is the recommended loss function for the current model.
    """
    
    def __init__(self, 
                 tversky_weight=0.4, 
                 focal_weight=0.3, 
                 bce_weight=0.3,
                 tversky_alpha=0.3, 
                 tversky_beta=0.7,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 pos_weight_multiplier=100.0):
        super(CombinedMedicalLoss, self).__init__()
        
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.focal_loss = FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta, gamma=focal_gamma)
        self.bce_loss = WeightedBCELoss(pos_weight_multiplier=pos_weight_multiplier)
        
    def forward(self, pred, target):
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        total_loss = (self.tversky_weight * tversky + 
                     self.focal_weight * focal + 
                     self.bce_weight * bce)
        
        return total_loss, {
            'tversky': tversky.item(),
            'focal': focal.item(),
            'bce': bce.item(),
            'total': total_loss.item()
        }

# Factory function for easy loss selection
def get_loss_function(loss_type='combined', **kwargs):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type: 'combined', 'tversky', 'focal_tversky', 'dice_focal', 'weighted_bce'
        **kwargs: Additional parameters for the loss function
    """
    loss_functions = {
        'combined': CombinedMedicalLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'dice_focal': DiceFocalLoss,
        'weighted_bce': WeightedBCELoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)

# Quick test function
def test_loss_functions():
    """Test all loss functions with dummy data."""
    # Create dummy data simulating extreme class imbalance
    pred = torch.rand(2, 1, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    # Add a small nodule (similar to real scenario)
    target[0, 0, 30:34, 30:34] = 1.0
    target[1, 0, 20:23, 40:43] = 1.0
    
    print("Testing loss functions with simulated lung nodule data:")
    print(f"Positive pixel ratio: {target.sum() / target.numel():.6f}")
    
    losses = ['tversky', 'focal_tversky', 'dice_focal', 'weighted_bce', 'combined']
    
    for loss_name in losses:
        loss_fn = get_loss_function(loss_name)
        if loss_name == 'combined':
            loss_val, loss_dict = loss_fn(pred, target)
            print(f"{loss_name}: {loss_val:.4f} {loss_dict}")
        else:
            loss_val = loss_fn(pred, target)
            print(f"{loss_name}: {loss_val:.4f}")

if __name__ == "__main__":
    test_loss_functions()
