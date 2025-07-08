"""
Configuration file for the DRR segmentation pipeline.
"""

import torch
import os

class Config:
    """Configuration class for training and inference."""
    
    # Data configuration
    DATA_ROOT = '../../../DRR dataset/LIDC_LDRI'
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 4
    TRAIN_SPLIT = 0.8
    AUGMENT_DATA = True
    NORMALIZE_DATA = True
    
    # Model configuration
    ALPHA = 0.5  # Attention modulation factor
    PRETRAINED_WEIGHTS = "resnet50-res512-all"
    
    # Training configuration
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    LAMBDA_ATTN = 0.5  # Weight for attention loss
    PATIENCE = 5  # Early stopping patience
    
    # Loss configuration
    DICE_WEIGHT = 0.6
    FOCAL_WEIGHT = 0.4
    USE_FOCAL_LOSS = True
    
    # Optimization configuration
    OPTIMIZER = 'adam'
    SCHEDULER = 'reduce_on_plateau'
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    WEIGHT_DECAY = 1e-5
    
    # Hardware configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Paths configuration
    SAVE_DIR = './checkpoints'
    LOG_DIR = './logs'
    RESULTS_DIR = './results'
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 1  # Save checkpoint every N epochs
    
    # Validation configuration
    VAL_INTERVAL = 1  # Validate every N epochs
    SAVE_BEST_ONLY = True
    METRIC_FOR_BEST = 'val_loss'  # 'val_loss', 'dice', 'iou'
    
    # Inference configuration
    INFERENCE_THRESHOLD = 0.5
    TTA = False  # Test Time Augmentation
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
    
    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration."""
        return {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY
        }
    
    @classmethod
    def get_scheduler_config(cls):
        """Get scheduler configuration."""
        if cls.SCHEDULER == 'reduce_on_plateau':
            return {
                'mode': 'min',
                'factor': cls.SCHEDULER_FACTOR,
                'patience': cls.SCHEDULER_PATIENCE,
                'verbose': True
            }
        elif cls.SCHEDULER == 'cosine':
            return {
                'T_max': cls.NUM_EPOCHS,
                'eta_min': 1e-6
            }
        return {}
    
    @classmethod
    def get_dataloader_config(cls):
        """Get dataloader configuration."""
        return {
            'batch_size': cls.BATCH_SIZE,
            'num_workers': cls.NUM_WORKERS,
            'pin_memory': cls.PIN_MEMORY,
            'persistent_workers': cls.NUM_WORKERS > 0
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                if not attr_name.startswith('get_') and not attr_name.startswith('create_') and not attr_name.startswith('print_'):
                    print(f"{attr_name}: {attr_value}")
        
        print("=" * 50)

# Development configuration (override for development)
class DevConfig(Config):
    """Development configuration with faster settings."""
    NUM_EPOCHS = 5
    BATCH_SIZE = 2
    LOG_INTERVAL = 5
    PATIENCE = 2

# Production configuration (override for production)
class ProdConfig(Config):
    """Production configuration with optimal settings."""
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    PATIENCE = 10
    AUGMENT_DATA = True
