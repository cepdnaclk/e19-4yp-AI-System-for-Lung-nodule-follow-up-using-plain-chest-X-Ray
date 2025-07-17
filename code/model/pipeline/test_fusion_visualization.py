"""
Quick test script to verify fusion visualization functionality.
"""

import sys
import os
sys.path.append('.')

def test_fusion_imports():
    """Test if all required imports work."""
    try:
        from visualize_model_internals import ModelInternalVisualizer
        from small_dataset_config import SmallDatasetConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_fusion_creation():
    """Test if fusion visualization methods can be created."""
    try:
        from visualize_model_internals import ModelInternalVisualizer
        from small_dataset_config import SmallDatasetConfig
        import numpy as np
        
        config = SmallDatasetConfig()
        
        # Create a mock visualizer (without loading model)
        visualizer = ModelInternalVisualizer(model_path=None, config=config)
        
        # Test fusion methods with dummy data
        dummy_xray = np.random.rand(512, 512)
        dummy_drr = np.random.rand(512, 512)
        dummy_attention = np.random.rand(256, 256)  # Different size to test resizing
        
        # Test all fusion methods
        weighted_fusion = visualizer._create_weighted_fusion(dummy_xray, dummy_drr, dummy_attention)
        overlay_fusion = visualizer._create_overlay_fusion(dummy_xray, dummy_drr, dummy_attention)
        rgb_fusion = visualizer._create_rgb_fusion(dummy_xray, dummy_drr, dummy_attention)
        
        print("✓ Weighted fusion created:", weighted_fusion.shape)
        print("✓ Overlay fusion created:", overlay_fusion.shape)
        print("✓ RGB fusion created:", rgb_fusion.shape)
        
        # Verify shapes are correct
        assert weighted_fusion.shape == dummy_xray.shape, "Weighted fusion shape mismatch"
        assert overlay_fusion.shape == dummy_xray.shape, "Overlay fusion shape mismatch"
        assert rgb_fusion.shape == (dummy_xray.shape[0], dummy_xray.shape[1], 3), "RGB fusion shape mismatch"
        
        print("✓ All fusion methods working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Fusion creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_availability():
    """Test if dataset is available for visualization."""
    try:
        from drr_dataset_loading import DRRDataset
        from small_dataset_config import SmallDatasetConfig
        
        config = SmallDatasetConfig()
        
        # Try to load a small validation dataset
        val_dataset = DRRDataset(
            data_root=config.DATA_ROOT,
            image_size=config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=config.NORMALIZE_DATA
        )
        
        print(f"✓ Dataset loaded: {len(val_dataset)} validation samples")
        print(f"✓ Data root: {config.DATA_ROOT}")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        print("  This is expected if the dataset path is not configured correctly")
        return False

if __name__ == "__main__":
    print("Testing Fusion Visualization Functionality")
    print("=" * 50)
    
    # Run tests
    import_success = test_fusion_imports()
    fusion_success = test_fusion_creation()
    dataset_success = test_dataset_availability()
    
    print("\\n" + "=" * 50)
    print("Test Results:")
    print(f"  Imports: {'✓ PASS' if import_success else '✗ FAIL'}")
    print(f"  Fusion Methods: {'✓ PASS' if fusion_success else '✗ FAIL'}")
    print(f"  Dataset: {'✓ PASS' if dataset_success else '✗ FAIL (expected if dataset not configured)'}")
    
    if import_success and fusion_success:
        print("\\n🎉 Fusion visualization functionality is ready!")
        print("\\nYou can now use the interactive visualizer with:")
        print("  python interactive_visualizer.py")
        print("\\nThe following new fusion visualizations will be generated:")
        print("  • overview_sample_X_Y.png (3x3 grid with fusion methods)")
        print("  • fusion_analysis_sample_X_Y.png (detailed fusion comparison)")
        print("  • fusion_quality_analysis_sample_X_Y.png (quality metrics)")
    else:
        print("\\n❌ Some tests failed. Please check the error messages above.")
