"""
Configuration optimized for small datasets (~600 samples).
Focuses on preventing overfitting and stable training.
"""

import torch
import os

class SmallDatasetConfig:
    """Configuration optimized for small medical datasets."""
    
    # Data configuration
    DATA_ROOT = '../../../DRR dataset/LIDC_LDRI'
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 2  # Smaller batch size for small datasets
    TRAIN_SPLIT = 0.8
    AUGMENT_DATA = True  # Important for small datasets
    NORMALIZE_DATA = True
    
    # Model configuration - lightweight
    ALPHA = 0.2  # Moderate attention influence
    PRETRAINED_WEIGHTS = "resnet50-res512-all"
    USE_CHANNEL_ATTENTION = False  # Disable to reduce complexity
    MODEL_TYPE = 'lightweight'  # 'lightweight' or 'minimal'
    
    # Training configuration - conservative
    LEARNING_RATE = 1e-4  # Higher learning rate for small datasets
    NUM_EPOCHS = 50  # More epochs needed for small datasets
    LAMBDA_ATTN = 0.05  # Very light attention supervision
    PATIENCE = 15  # More patience for small datasets
    
    # Loss configuration - simplified
    LOSS_TYPE = 'small_dataset'
    DICE_WEIGHT = 0.6
    BCE_WEIGHT = 0.4
    POS_WEIGHT = 50.0  # Moderate weighting to avoid instability
    
    # Optimization configuration - conservative
    OPTIMIZER = 'adam'  # Simple Adam optimizer
    SCHEDULER = 'reduce_on_plateau'
    SCHEDULER_FACTOR = 0.7  # Less aggressive reduction
    SCHEDULER_PATIENCE = 5
    WEIGHT_DECAY = 1e-3  # Strong regularization for small datasets
    
    # Hardware configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Reduce for stability
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Paths configuration
    SAVE_DIR = './checkpoints_lightweight'
    LOG_DIR = './logs_lightweight'
    RESULTS_DIR = './results_lightweight'
    
    # Evaluation configuration
    THRESHOLD_OPTIMIZATION = True
    VISUALIZATION_FREQUENCY = 10  # Less frequent to save time
    
    # Data augmentation - important for small datasets
    ROTATION_DEGREES = 10  # Moderate augmentation
    BRIGHTNESS_FACTOR = 0.1
    CONTRAST_FACTOR = 0.1
    HORIZONTAL_FLIP_PROB = 0.3
    
    # Training options - conservative
    GRADIENT_CLIP_NORM = 0.5  # Strong gradient clipping
    MIXED_PRECISION = False  # Avoid for stability
    EARLY_STOPPING_MIN_DELTA = 1e-4  # Sensitive to small improvements
    
    # Validation configuration
    VAL_INTERVAL = 1
    SAVE_BEST_ONLY = True
    METRIC_FOR_BEST = 'dice'  # Focus on Dice score
    
    # Inference configuration
    INFERENCE_THRESHOLD = 0.5
    TTA = False
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
    
    @classmethod
    def get_model_params(cls):
        """Get model parameters based on dataset size."""
        return {
            'alpha': cls.ALPHA,
            'freeze_early_layers': True,  # Heavy freezing for small datasets
            'model_type': cls.MODEL_TYPE
        }
    
    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration."""
        return {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY
        }
    
    def __repr__(self):
        """String representation of config."""
        return f"SmallDatasetConfig(batch_size={self.BATCH_SIZE}, lr={self.LEARNING_RATE}, model={self.MODEL_TYPE})"
