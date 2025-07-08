# DRR Segmentation Pipeline

This pipeline implements an enhanced deep learning model for lung nodule segmentation using DRR (Digitally Reconstructed Radiograph) images and attention mechanisms.

## Features

### ✨ **Key Improvements Made**

1. **Enhanced Model Architecture**
   - Improved attention mechanism with batch normalization and dropout
   - Progressive upsampling in segmentation head
   - Feature fusion module for better representation learning

2. **Robust Training Pipeline**
   - Train/validation split with proper evaluation
   - Learning rate scheduling
   - Early stopping and model checkpointing
   - Comprehensive logging and error handling

3. **Advanced Data Loading**
   - Data augmentation with synchronized transforms
   - Robust error handling for corrupted files
   - Dataset statistics and validation

4. **Comprehensive Evaluation**
   - Multiple metrics (Dice, IoU, Precision, Recall, F1)
   - Visual evaluation with sample predictions
   - Detailed result saving and analysis

5. **Improved Loss Functions**
   - Combined Dice + Focal loss
   - Attention supervision loss
   - Configurable loss weights

## Project Structure

```
pipeline/
├── main.py                    # Main training script (enhanced)
├── model.py                   # Enhanced model architecture
├── drr_dataset_loading.py     # Improved dataset loading
├── util.py                    # Enhanced utilities and metrics
├── config.py                  # Configuration management
├── evaluate.py                # Comprehensive evaluation script
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is structured as:
```
DRR dataset/
└── LIDC_LDRI/
    ├── 1-200/
    │   ├── *_drr.png
    │   └── *_drr_mask.png
    ├── 201-400/
    └── ...
```

## Usage

### Training

Run the main training script:
```bash
python main.py
```

The script will:
- Load and validate the dataset
- Split into train/validation sets
- Train the model with proper logging
- Save the best model checkpoint
- Generate training curves and sample predictions

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model_path ./checkpoints/best_model.pth --config prod
```

### Configuration

Modify `config.py` to adjust:
- Training hyperparameters
- Model architecture settings
- Data augmentation options
- Paths and directories

Available configurations:
- `Config`: Default configuration
- `DevConfig`: Fast settings for development
- `ProdConfig`: Optimal settings for production

## Key Improvements Explained

### 1. **Model Architecture (`model.py`)**

**Before:**
- Simple 2-layer attention network
- Basic segmentation head
- Single upsampling step

**After:**
- Enhanced attention module with batch normalization
- Progressive upsampling with feature refinement
- Feature fusion for better representation learning
- Dropout for regularization

### 2. **Training Pipeline (`main.py`)**

**Before:**
- No validation split
- Basic training loop
- Limited error handling
- No model saving

**After:**
- Proper train/validation split
- Learning rate scheduling
- Comprehensive logging
- Model checkpointing with best model saving
- Training curve visualization
- Error handling and recovery

### 3. **Data Loading (`drr_dataset_loading.py`)**

**Before:**
- Basic image loading
- No error handling
- No augmentation

**After:**
- Robust file validation
- Data augmentation with synchronized transforms
- Enhanced error handling
- Dataset statistics
- Configurable normalization

### 4. **Evaluation (`evaluate.py`)**

**Before:**
- No dedicated evaluation script

**After:**
- Comprehensive metric calculation
- Visual evaluation with sample predictions
- Detailed result saving
- Best/worst sample analysis
- Statistical reporting

### 5. **Loss Functions (`util.py`)**

**Before:**
- Simple dice loss only

**After:**
- Combined Dice + Focal loss
- Attention supervision
- Multiple evaluation metrics
- Enhanced visualization

## Configuration Options

Key parameters you can adjust in `config.py`:

```python
# Training
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 4

# Model
ALPHA = 0.5  # Attention strength
LAMBDA_ATTN = 0.5  # Attention loss weight

# Loss
DICE_WEIGHT = 0.6
FOCAL_WEIGHT = 0.4

# Data
AUGMENT_DATA = True
TRAIN_SPLIT = 0.8
```

## Output Structure

After training and evaluation:

```
pipeline/
├── checkpoints/
│   ├── best_model.pth
│   └── training_curves.png
├── logs/
│   └── training.log
└── results/
    ├── evaluation_metrics.txt
    ├── sample_predictions.png
    ├── best_sample_1.png
    └── worst_sample_1.png
```

## Monitoring Training

The enhanced pipeline provides:

1. **Real-time logging**: Progress updates every 10 batches
2. **Training curves**: Automatic generation of loss plots
3. **Validation metrics**: Regular validation during training
4. **Early stopping**: Prevents overfitting
5. **Model checkpointing**: Saves best performing model

## Performance Metrics

The evaluation script calculates:

- **Dice Coefficient**: Measure of overlap between prediction and ground truth
- **IoU (Jaccard)**: Intersection over Union
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce `BATCH_SIZE` in config
2. **Dataset not found**: Check `DATA_ROOT` path in config
3. **Slow training**: Enable GPU and increase `NUM_WORKERS`

### Performance Tips:

1. Use GPU for faster training
2. Adjust batch size based on available memory
3. Enable data augmentation for better generalization
4. Monitor validation loss to prevent overfitting

## Future Enhancements

Potential improvements:
1. Multi-scale training and inference
2. Ensemble methods
3. Advanced augmentation strategies
4. Transfer learning from other medical imaging tasks
5. Integration with medical imaging frameworks

## License

This project is part of the AI System for Lung nodule follow-up research.
