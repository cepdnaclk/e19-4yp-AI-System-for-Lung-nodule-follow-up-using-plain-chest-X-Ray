"""
Simplified loss functions optimized for small datasets.
Focuses on stability and preventing overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiceLoss(nn.Module):
    """Simple, stable Dice loss for small datasets."""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class StableWeightedBCE(nn.Module):
    """Stable weighted BCE loss for small datasets."""
    
    def __init__(self, pos_weight=50.0):  # Moderate weighting
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        # Clamp predictions to prevent numerical instability
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Simple weighted BCE
        loss = -(self.pos_weight * target * torch.log(pred) + 
                (1 - target) * torch.log(1 - pred))
        
        return loss.mean()

class SmallDatasetLoss(nn.Module):
    """
    Combined loss optimized for small datasets.
    Balances simplicity with effectiveness.
    """
    
    def __init__(self, dice_weight=0.6, bce_weight=0.4, pos_weight=50.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = SimpleDiceLoss()
        self.bce_loss = StableWeightedBCE(pos_weight=pos_weight)
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        
        return total_loss, {
            'dice': dice.item(),
            'bce': bce.item(),
            'total': total_loss.item()
        }

def get_small_dataset_loss(pos_weight=50.0):
    """Factory function for small dataset loss."""
    return SmallDatasetLoss(pos_weight=pos_weight)
