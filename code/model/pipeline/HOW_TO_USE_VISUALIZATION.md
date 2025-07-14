# How to Use visualization.py - Complete Guide

## 🎯 Overview

The `visualization.py` file provides powerful tools to visualize and debug your lung nodule segmentation model. It helps you understand:
- How the attention mechanism works
- Prediction quality across different samples
- Optimal threshold selection
- Feature map analysis
- Dataset statistics

## 🚀 Quick Start

### **Step 1: Basic Setup**

```python
import torch
from torch.utils.data import DataLoader
from visualization import ModelVisualizer, visualize_dataset_statistics
from drr_dataset_loading import DRRSegmentationDataset
from custom_model import ImprovedXrayDRRSegmentationModel
import torchxrayvision as xrv

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Method 1: Load from checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')
pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
model = ImprovedXrayDRRSegmentationModel(pretrained_model)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Method 2: Use your existing trained model
# model = your_trained_model

# Create visualizer
visualizer = ModelVisualizer(model, device)
```

### **Step 2: Load Data**

```python
# Load dataset
dataset = DRRSegmentationDataset(
    root_dir='../../../DRR dataset/LIDC_LDRI',
    image_size=(512, 512)
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Get a sample batch
sample_batch = next(iter(dataloader))
xray = sample_batch["xray"].to(device)
drr = sample_batch["drr"].to(device)  
mask = sample_batch["mask"].to(device)
```

## 🔍 Visualization Functions

### **1. Attention Mechanism Visualization**

```python
# Visualize how attention works on a single sample
visualizer.visualize_attention_mechanism(
    xray=xray, 
    drr=drr, 
    mask=mask, 
    save_path='results/attention_analysis.png'
)
```

**What it shows:**
- Input X-ray image
- Input DRR image  
- Generated attention map
- Ground truth mask
- Model prediction
- X-ray with attention overlay

**Use case:** Check if the model's attention is focusing on actual nodule locations.

### **2. Prediction Quality Analysis**

```python
# Analyze prediction quality across multiple samples
metrics = visualizer.analyze_prediction_quality(
    dataloader=dataloader,
    num_samples=20,
    save_path='results/quality_analysis.png'
)

# Print summary statistics
print("Quality Analysis Results:")
for metric, values in metrics.items():
    print(f"{metric.title()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
```

**What it shows:**
- Grid of sample predictions vs ground truth
- Dice and IoU scores for each sample
- Statistical summary of performance

**Use case:** Get overall sense of model performance and identify challenging cases.

### **3. Threshold Optimization**

```python
# Find optimal threshold for your model
metrics, optimal_threshold = visualizer.threshold_analysis(
    dataloader=dataloader,
    save_path='results/threshold_analysis.png'
)

print(f"Recommended threshold: {optimal_threshold:.3f}")
```

**What it shows:**
- Dice, IoU, Precision, Recall vs threshold curves
- Optimal threshold recommendation
- Performance trade-offs

**Use case:** Determine the best threshold for converting probabilities to binary predictions.

### **4. Feature Map Visualization**

```python
# Visualize intermediate feature representations
visualizer.visualize_feature_maps(
    xray=xray,
    drr=drr,
    layer_name='layer4',  # Which layer to visualize
    save_path='results/feature_maps.png'
)
```

**What it shows:**
- 16 feature maps from the specified layer
- What patterns the model has learned to detect

**Use case:** Debug what the model is learning at different depths.

### **5. Dataset Statistics**

```python
# Visualize dataset characteristics
visualize_dataset_statistics(
    dataset=dataset,
    save_path='results/dataset_stats.png'
)
```

**What it shows:**
- Distribution of positive pixel ratios
- Image intensity distributions
- Sample images and masks

**Use case:** Understand your data distribution and potential biases.

## 🛠️ Complete Example Script

Create a file `run_visualization.py`:

```python
"""
Complete example of using visualization tools.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchxrayvision as xrv

# Import your modules
from visualization import ModelVisualizer, visualize_dataset_statistics
from drr_dataset_loading import DRRSegmentationDataset
from custom_model import ImprovedXrayDRRSegmentationModel

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = DRRSegmentationDataset(
        root_dir='../../../DRR dataset/LIDC_LDRI',
        image_size=(512, 512)
    )
    
    # Create validation split (use same split as training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Load trained model
    try:
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create visualizer
    visualizer = ModelVisualizer(model, device)
    
    # Get sample data
    sample_batch = next(iter(val_loader))
    xray = sample_batch["xray"].to(device)
    drr = sample_batch["drr"].to(device)
    mask = sample_batch["mask"].to(device)
    
    print("🔍 Starting visualization analysis...")
    
    # 1. Attention mechanism
    print("1. Analyzing attention mechanism...")
    visualizer.visualize_attention_mechanism(
        xray=xray, drr=drr, mask=mask,
        save_path='results/attention_analysis.png'
    )
    
    # 2. Prediction quality  
    print("2. Analyzing prediction quality...")
    metrics = visualizer.analyze_prediction_quality(
        dataloader=val_loader,
        num_samples=16,
        save_path='results/quality_analysis.png'
    )
    
    # 3. Threshold analysis
    print("3. Optimizing threshold...")
    threshold_metrics, optimal_threshold = visualizer.threshold_analysis(
        dataloader=val_loader,
        save_path='results/threshold_analysis.png'
    )
    
    # 4. Feature maps
    print("4. Visualizing feature maps...")
    visualizer.visualize_feature_maps(
        xray=xray, drr=drr,
        layer_name='layer4',
        save_path='results/feature_maps.png'
    )
    
    # 5. Dataset statistics
    print("5. Analyzing dataset statistics...")
    visualize_dataset_statistics(
        dataset=dataset,
        save_path='results/dataset_stats.png'
    )
    
    print("✅ All visualizations completed!")
    print("📁 Check the 'results/' folder for output images")
    print(f"🎯 Recommended threshold: {optimal_threshold:.3f}")

if __name__ == "__main__":
    main()
```

## 📊 How to Run

### **Option 1: Run the complete analysis**
```bash
cd code/model/pipeline
python run_visualization.py
```

### **Option 2: Interactive analysis in Jupyter**
```python
# In a Jupyter notebook
%matplotlib inline

# Load your model and data (as shown above)
# Then run individual visualization functions

# Quick attention check
visualizer.visualize_attention_mechanism(xray, drr, mask)

# Quick quality check  
metrics = visualizer.analyze_prediction_quality(val_loader, num_samples=8)
```

### **Option 3: Integrate with training**
```python
# Add to your training script
from visualization import ModelVisualizer

# After each epoch or at the end of training
if epoch % 5 == 0:  # Every 5 epochs
    visualizer = ModelVisualizer(model, device)
    visualizer.visualize_attention_mechanism(
        xray, drr, mask,
        save_path=f'results/attention_epoch_{epoch}.png'
    )
```

## 🎯 What to Look For

### **Good Attention Maps:**
- ✅ Bright spots on actual nodule locations
- ✅ Low attention on background/healthy tissue
- ✅ Attention boundaries align with ground truth

### **Poor Attention Maps:**
- ❌ Random/uniform attention across image
- ❌ High attention on obviously healthy areas
- ❌ No correlation with ground truth masks

### **Good Predictions:**
- ✅ Dice scores > 0.5
- ✅ Predictions match ground truth shape
- ✅ Minimal false positives

### **Signs of Problems:**
- ❌ All predictions are black (model predicts no nodules)
- ❌ All predictions are white (model predicts everything)
- ❌ Dice scores < 0.1

## 🚨 Troubleshooting

### **Error: Model not compatible**
```python
# If you get attribute errors, check your model type
if hasattr(model, 'spatial_attention'):
    print("✅ Using improved model")
else:
    print("⚠️ Using old model - some features may not work")
```

### **Error: CUDA out of memory**
```python
# Reduce batch size in visualizations
visualizer.analyze_prediction_quality(
    dataloader=val_loader,
    num_samples=8,  # Reduce from 20
    save_path='results/quality_analysis.png'
)
```

### **Error: No module found**
```bash
# Make sure you're in the right directory
cd code/model/pipeline

# Check if files exist
ls -la visualization.py
ls -la custom_model.py
```

## 📈 Interpreting Results

### **Attention Quality:**
- **Good**: Attention focuses on nodule regions with >0.7 intensity
- **Moderate**: Some attention on nodules but also background noise
- **Poor**: Random or no attention correlation with ground truth

### **Prediction Quality:**
- **Excellent**: Dice > 0.7, IoU > 0.5
- **Good**: Dice 0.4-0.7, IoU 0.3-0.5  
- **Poor**: Dice < 0.3, IoU < 0.2

### **Threshold Selection:**
- **Optimal**: Usually between 0.3-0.7 for medical images
- **Too High**: High precision but low recall (misses nodules)
- **Too Low**: High recall but low precision (many false positives)

Now you can fully analyze and debug your lung nodule segmentation model! 🔬✨
