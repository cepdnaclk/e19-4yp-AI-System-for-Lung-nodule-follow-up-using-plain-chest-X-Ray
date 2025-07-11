"""
Simple test script to check all imports are working correctly.
"""

import sys
import os

# Test basic imports
try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torchxrayvision as xrv
    print("✓ TorchXRayVision imported successfully")
except ImportError as e:
    print(f"✗ TorchXRayVision import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

# Test local imports
try:
    from config_ import Config
    print("✓ Config imported successfully")
    config = Config()
    print(f"  Device: {config.DEVICE}")
    print(f"  Data root: {config.DATA_ROOT}")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from custom_model import ImprovedXrayDRRSegmentationModel, XrayDRRSegmentationModel
    print("✓ Custom models imported successfully")
except Exception as e:
    print(f"✗ Custom models import failed: {e}")

try:
    from util import dice_loss, hybrid_loss, calculate_metrics, focal_loss
    print("✓ Util functions imported successfully")
except Exception as e:
    print(f"✗ Util functions import failed: {e}")

try:
    from drr_dataset_loading import DRRSegmentationDataset
    print("✓ Dataset loading imported successfully")
except Exception as e:
    print(f"✗ Dataset loading import failed: {e}")

try:
    from visualization import ModelVisualizer, visualize_dataset_statistics
    print("✓ Visualization functions imported successfully")
except Exception as e:
    print(f"✗ Visualization functions import failed: {e}")

# Test model instantiation
try:
    print("\nTesting model instantiation...")
    pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
    print("✓ Pretrained model loaded successfully")
    
    model = ImprovedXrayDRRSegmentationModel(pretrained_model, alpha=0.3)
    print("✓ Improved model instantiated successfully")
    
    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_xray = torch.randn(1, 1, 512, 512).to(device)
    dummy_drr = torch.randn(1, 1, 512, 512).to(device)
    
    result = model(dummy_xray, dummy_drr)
    if isinstance(result, dict):
        print(f"✓ Forward pass successful - Segmentation shape: {result['segmentation'].shape}")
        if 'attention' in result:
            print(f"  Attention shape: {result['attention'].shape}")
    else:
        print(f"✓ Forward pass successful - Output shape: {result.shape}")
        
except Exception as e:
    print(f"✗ Model testing failed: {e}")
    import traceback
    traceback.print_exc()

print("\nImport test completed!")
