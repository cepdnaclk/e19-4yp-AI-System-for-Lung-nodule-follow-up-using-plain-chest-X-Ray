"""
Quick visualization test script - minimal setup for immediate testing.
Use this to quickly check if your model and visualization tools work.
"""

import torch
import os
import matplotlib.pyplot as plt

def quick_test():
    """Quick test of visualization functionality."""
    
    print("🚀 Quick Visualization Test")
    print("=" * 40)
    
    # Check if necessary files exist
    required_files = [
        'visualization.py',
        'custom_model.py', 
        'drr_dataset_loading.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("💡 Make sure you're in the pipeline directory")
        return False
    
    print("✅ All required files found")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    # Check if model checkpoint exists
    if os.path.exists('checkpoints/best_model.pth'):
        print("✅ Model checkpoint found")
    else:
        print("⚠️ No model checkpoint found")
        print("💡 Train the model first with: python improved_train.py")
        return False
    
    # Check if dataset exists
    dataset_path = '../../../DRR dataset/LIDC_LDRI'
    if os.path.exists(dataset_path):
        print("✅ Dataset found")
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Update the dataset path in the script")
        return False
    
    print("\n🎯 Everything looks good! You can now run:")
    print("   python run_visualization.py")
    print("   or")
    print("   python run_visualization.py --quick")
    
    return True

def test_imports():
    """Test if all modules can be imported."""
    
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import torchxrayvision as xrv
        print("✅ TorchXRayVision")
    except ImportError as e:
        print(f"❌ TorchXRayVision: {e}")
        print("💡 Install with: pip install torchxrayvision")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
        
    try:
        from visualization import ModelVisualizer
        print("✅ Visualization module")
    except ImportError as e:
        print(f"❌ Visualization module: {e}")
        return False
        
    try:
        from custom_model import ImprovedXrayDRRSegmentationModel
        print("✅ Custom model")
    except ImportError as e:
        print(f"❌ Custom model: {e}")
        return False
        
    try:
        from drr_dataset_loading import DRRSegmentationDataset
        print("✅ Dataset loader")
    except ImportError as e:
        print(f"❌ Dataset loader: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def main():
    """Main test function."""
    
    print("🧪 Visualization Setup Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import test failed")
        return
    
    print()
    
    # Test setup
    if not quick_test():
        print("\n❌ Setup test failed")
        return
    
    print("\n🎉 All tests passed!")
    print("\n📋 Next steps:")
    print("1. Run full analysis: python run_visualization.py")
    print("2. Quick attention check: python run_visualization.py --quick")
    print("3. Check results in the 'results/' folder")

if __name__ == "__main__":
    main()
