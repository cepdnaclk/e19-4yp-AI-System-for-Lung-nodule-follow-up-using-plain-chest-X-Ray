"""
Simple and clean configuration for the improved model.
"""

import torch

class SimpleConfig:
    """Simplified configuration class with essential parameters only."""
    
    # Data settings
    DATA_ROOT = '../../../DRR dataset/LIDC_LDRI'
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 4
    TRAIN_SPLIT = 0.8
    
    # Model settings
    ALPHA = 0.3  # Attention modulation factor (reduced from 0.5)
    PRETRAINED_WEIGHTS = "resnet50-res512-all"
    
    # Training settings
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    PATIENCE = 5  # Early stopping patience
    
    # Loss settings
    DICE_WEIGHT = 0.7
    FOCAL_WEIGHT = 0.3
    ATTENTION_WEIGHT = 0.1  # Reduced attention supervision
    
    # Hardware settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Paths
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary."""
        return {
            'data_root': cls.DATA_ROOT,
            'image_size': cls.IMAGE_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'train_split': cls.TRAIN_SPLIT,
            'alpha': cls.ALPHA,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'patience': cls.PATIENCE,
            'dice_weight': cls.DICE_WEIGHT,
            'focal_weight': cls.FOCAL_WEIGHT,
            'attention_weight': cls.ATTENTION_WEIGHT,
            'device': cls.DEVICE,
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
            'save_dir': cls.SAVE_DIR,
            'log_dir': cls.LOG_DIR
        }
