"""
Training configuration for improved model with supervised attention.
This addresses the attention-segmentation alignment issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Updated training configuration
IMPROVED_CONFIG = {
    # Model parameters
    'MODEL_TYPE': 'improved',  # Use improved model
    'ALPHA': 0.3,  # Increased attention influence
    'USE_SUPERVISED_ATTENTION': True,  # Enable attention supervision
    'TARGET_PATHOLOGY': 'Nodule',
    
    # Training parameters
    'BATCH_SIZE': 2,
    'LEARNING_RATE': 5e-5,  # Slightly reduced for stability
    'WEIGHT_DECAY': 1e-4,
    'NUM_EPOCHS': 150,
    
    # Loss weights
    'SEGMENTATION_WEIGHT': 1.0,
    'ATTENTION_WEIGHT': 0.5,  # Weight for attention supervision loss
    'DICE_WEIGHT': 0.3,
    'BCE_WEIGHT': 0.7,
    
    # Training strategy
    'WARMUP_EPOCHS': 10,  # Epochs to train only attention supervision
    'ATTENTION_SUPERVISION_EPOCHS': 50,  # Epochs to use attention supervision
    'SCHEDULER_PATIENCE': 15,
    'EARLY_STOPPING_PATIENCE': 25,
    
    # Data augmentation
    'AUGMENT_TRAINING': True,
    'NORMALIZE_IMAGES': True,
    
    # Validation
    'VALIDATION_FREQUENCY': 5,
    'SAVE_BEST_MODEL': True,
    
    # Logging
    'LOG_FREQUENCY': 10,
    'SAVE_ATTENTION_MAPS': True,
    'ATTENTION_SAVE_FREQUENCY': 20,
}

class ImprovedCombinedLoss(nn.Module):
    """Improved loss function that includes attention supervision."""
    
    def __init__(self, 
                 seg_weight=1.0, 
                 attention_weight=0.5,
                 dice_weight=0.3, 
                 bce_weight=0.7,
                 use_focal_loss=True):
        super().__init__()
        self.seg_weight = seg_weight
        self.attention_weight = attention_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.use_focal_loss = use_focal_loss
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for segmentation."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for hard examples."""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def attention_alignment_loss(self, attention, segmentation, target):
        """Loss to encourage attention-segmentation alignment."""
        # Resize attention to match segmentation if needed
        if attention.shape != segmentation.shape:
            attention_resized = F.interpolate(
                attention, 
                size=segmentation.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            attention_resized = attention
        
        # Convert segmentation to attention-like format
        seg_prob = torch.sigmoid(segmentation)
        
        # Compute alignment loss (MSE between attention and segmentation probability)
        alignment_loss = F.mse_loss(attention_resized, seg_prob)
        
        return alignment_loss
    
    def forward(self, predictions, targets, epoch=0, total_epochs=100):
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with 'segmentation', 'attention', and optionally 'attention_loss'
            targets: Ground truth masks
            epoch: Current epoch for dynamic weighting
            total_epochs: Total number of epochs
        """
        segmentation = predictions['segmentation']
        attention = predictions['attention']
        
        # Segmentation losses
        if self.use_focal_loss:
            bce_loss = self.focal_loss(segmentation, targets)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(segmentation, targets)
        
        dice_loss = self.dice_loss(segmentation, targets)
        seg_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        total_loss = self.seg_weight * seg_loss
        
        loss_info = {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
        }
        
        # Add attention supervision loss if available
        if 'attention_loss' in predictions:
            attention_supervision_loss = predictions['attention_loss']
            
            # Dynamic weighting: more attention supervision early in training
            attention_weight = self.attention_weight * max(0.1, 1.0 - epoch / (total_epochs * 0.5))
            
            total_loss = total_loss + attention_weight * attention_supervision_loss
            loss_info['attention_supervision_loss'] = attention_supervision_loss
            loss_info['attention_weight'] = attention_weight
            loss_info['total_loss'] = total_loss
        
        # Add attention-segmentation alignment loss
        alignment_loss = self.attention_alignment_loss(attention, segmentation, targets)
        alignment_weight = 0.1  # Small weight for alignment
        
        total_loss = total_loss + alignment_weight * alignment_loss
        loss_info['alignment_loss'] = alignment_loss
        loss_info['total_loss'] = total_loss
        
        return total_loss, loss_info

def create_improved_model():
    """Create the improved model with better attention."""
    from improved_model import ImprovedXrayDRRModel
    
    model = ImprovedXrayDRRModel(
        alpha=IMPROVED_CONFIG['ALPHA'],
        use_supervised_attention=IMPROVED_CONFIG['USE_SUPERVISED_ATTENTION'],
        target_pathology=IMPROVED_CONFIG['TARGET_PATHOLOGY']
    )
    
    return model

def create_improved_loss():
    """Create the improved loss function."""
    return ImprovedCombinedLoss(
        seg_weight=IMPROVED_CONFIG['SEGMENTATION_WEIGHT'],
        attention_weight=IMPROVED_CONFIG['ATTENTION_WEIGHT'],
        dice_weight=IMPROVED_CONFIG['DICE_WEIGHT'],
        bce_weight=IMPROVED_CONFIG['BCE_WEIGHT'],
        use_focal_loss=True
    )

def get_improved_optimizer(model):
    """Create optimizer with different learning rates for different parts."""
    # Different learning rates for different components
    backbone_params = []
    attention_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            backbone_params.append(param)
        elif 'attention' in name:
            attention_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': IMPROVED_CONFIG['LEARNING_RATE'] * 0.1},  # Lower LR for backbone
        {'params': attention_params, 'lr': IMPROVED_CONFIG['LEARNING_RATE'] * 2.0},  # Higher LR for attention
        {'params': head_params, 'lr': IMPROVED_CONFIG['LEARNING_RATE']}  # Normal LR for head
    ], weight_decay=IMPROVED_CONFIG['WEIGHT_DECAY'])
    
    return optimizer

def get_improved_scheduler(optimizer):
    """Create learning rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=IMPROVED_CONFIG['SCHEDULER_PATIENCE'],
        verbose=True,
        min_lr=1e-7
    )

# Training strategy guidance
TRAINING_STRATEGY = {
    'phase_1': {
        'name': 'Attention Supervision Phase',
        'epochs': '1-50',
        'focus': 'Train attention mechanism with ground truth supervision',
        'attention_weight': 'High (0.5 -> 0.1)',
        'segmentation_weight': 'Normal (1.0)'
    },
    'phase_2': {
        'name': 'Joint Training Phase', 
        'epochs': '51-150',
        'focus': 'Fine-tune both attention and segmentation together',
        'attention_weight': 'Low (0.1 -> 0.05)',
        'segmentation_weight': 'High (1.0)'
    }
}

print("Improved Training Configuration Loaded:")
print(f"- Model: Improved XrayDRR with supervised attention")
print(f"- Alpha: {IMPROVED_CONFIG['ALPHA']}")
print(f"- Attention supervision: {IMPROVED_CONFIG['USE_SUPERVISED_ATTENTION']}")
print(f"- Training epochs: {IMPROVED_CONFIG['NUM_EPOCHS']}")
print(f"- Attention supervision for first {IMPROVED_CONFIG['ATTENTION_SUPERVISION_EPOCHS']} epochs")
