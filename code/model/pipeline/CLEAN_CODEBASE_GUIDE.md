# Clean Codebase Guide - Improved Model Training & Visualization

This directory contains the cleaned up, essential files for your improved model training and visualization workflow.

## 🎯 Core Files (Essential)

### Training & Model
- **`train_improved_simple.py`** - Main training script (this is what you use)
- **`improved_model.py`** - ImprovedXrayDRRModel architecture with supervised attention
- **`improved_config.py`** - Configuration for improved model training
- **`drr_dataset_loading.py`** - Dataset loading and preprocessing
- **`util.py`** - Utility functions (dice_coefficient, etc.)

### Visualization
- **`interactive_visualizer.py`** - Interactive tool for exploring model outputs
- **`visualize_model_internals.py`** - Core visualization functions and model analysis
- **`small_dataset_config.py`** - Base configuration (used by visualizers)

### Evaluation
- **`evaluate.py`** - Model evaluation scripts

## 🚀 Quick Start

### Training
```bash
python train_improved_simple.py
```

### Visualization
```bash
python interactive_visualizer.py
```

## 📁 Documentation
- **`ARCHITECTURE_IMPROVEMENTS.md`** - Details about model improvements
- **`ATTENTION_ALIGNMENT_SOLUTION.md`** - Solution for attention alignment issues
- **`HOW_TO_USE_VISUALIZATION.md`** - Visualization guide
- **`Nodule_Specific_Adaptation.md`** - Nodule-specific adaptations
- **`Small_Dataset_Solution.md`** - Small dataset handling strategies
- **`QUICK_START.md`** - Quick start guide

## 🗂️ Other
- **`requirements.txt`** - Dependencies
- **`archive_experimental/`** - Experimental code archive
- **Sample outputs**: `*.png` files showing example visualizations

## 🧹 Removed Files
The following unnecessary files have been removed to clean up the codebase:
- Old training scripts: `train.py`, `main.py`, `simple_train.py`, `improved_train.py`
- Old model files: `lightweight_model.py`, `custom_model.py`
- Old configs: `config_.py`, `lightweight_losses.py`
- Test files: `test_*.py`, `model_comparison.py`
- Old visualization: `visualization.py`, `run_visualization.py`
- Redundant files: `improved_visualizer.py`, `improved_losses.py`

## 🎯 Your Workflow
1. **Train**: `python train_improved_simple.py`
2. **Visualize**: `python interactive_visualizer.py`
3. **Evaluate**: `python evaluate.py` (if needed)

That's it! Clean and simple.
