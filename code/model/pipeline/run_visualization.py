"""
Complete example of using visualization tools for lung nodule segmentation model.
Run this script to generate comprehensive visualizations of your model's performance.
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torchxrayvision as xrv

# Import your modules
from visualization import ModelVisualizer, visualize_dataset_statistics
from drr_dataset_loading import DRRSegmentationDataset
from custom_model import ImprovedXrayDRRSegmentationModel
from config_ import Config

def main():
    """Main visualization analysis function."""
    
    # Setup
    config = Config()
    device = config.DEVICE
    print(f"🚀 Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("📊 Loading dataset...")
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config.DATA_ROOT,
            image_size=config.IMAGE_SIZE
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create validation split (use same split as training)
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    print(f"✅ Validation set: {len(val_dataset)} samples")
    
    # Load trained model
    print("🤖 Loading trained model...")
    try:
        checkpoint_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            print(f"❌ Model checkpoint not found at: {checkpoint_path}")
            print("💡 Make sure you have trained the model first using:")
            print("   python improved_train.py")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pretrained_model = xrv.models.ResNet(weights=config.PRETRAINED_WEIGHTS)
        model = ImprovedXrayDRRSegmentationModel(
            pretrained_model,
            alpha=config.ALPHA,
            use_channel_attention=config.USE_CHANNEL_ATTENTION
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"📈 Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create visualizer
    visualizer = ModelVisualizer(model, device)
    
    # Get sample data
    print("🔍 Preparing sample data...")
    try:
        sample_batch = next(iter(val_loader))
        xray = sample_batch["xray"].to(device)
        drr = sample_batch["drr"].to(device)
        mask = sample_batch["mask"].to(device)
        print(f"✅ Sample batch loaded: {xray.shape}")
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return
    
    print("\n🎨 Starting comprehensive visualization analysis...")
    print("=" * 60)
    
    # 1. Attention mechanism analysis
    print("1️⃣ Analyzing attention mechanism...")
    try:
        visualizer.visualize_attention_mechanism(
            xray=xray, drr=drr, mask=mask,
            save_path='results/attention_analysis.png'
        )
        print("   ✅ Attention analysis saved to results/attention_analysis.png")
    except Exception as e:
        print(f"   ❌ Error in attention analysis: {e}")
    
    # 2. Prediction quality analysis
    print("\n2️⃣ Analyzing prediction quality across samples...")
    try:
        metrics = visualizer.analyze_prediction_quality(
            dataloader=val_loader,
            num_samples=16,
            save_path='results/quality_analysis.png'
        )
        print("   ✅ Quality analysis saved to results/quality_analysis.png")
        
        # Print summary
        print("   📊 Performance Summary:")
        for metric, values in metrics.items():
            print(f"      {metric.title()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
    except Exception as e:
        print(f"   ❌ Error in quality analysis: {e}")
    
    # 3. Threshold optimization
    print("\n3️⃣ Optimizing prediction threshold...")
    try:
        threshold_metrics, optimal_threshold = visualizer.threshold_analysis(
            dataloader=val_loader,
            save_path='results/threshold_analysis.png'
        )
        print("   ✅ Threshold analysis saved to results/threshold_analysis.png")
        print(f"   🎯 Recommended threshold: {optimal_threshold:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in threshold analysis: {e}")
        optimal_threshold = 0.5  # Default fallback
    
    # 4. Feature map visualization
    print("\n4️⃣ Visualizing learned features...")
    try:
        visualizer.visualize_feature_maps(
            xray=xray, drr=drr,
            layer_name='layer4',
            save_path='results/feature_maps.png'
        )
        print("   ✅ Feature maps saved to results/feature_maps_layer4.png")
    except Exception as e:
        print(f"   ❌ Error in feature visualization: {e}")
    
    # 5. Dataset statistics
    print("\n5️⃣ Analyzing dataset characteristics...")
    try:
        visualize_dataset_statistics(
            dataset=dataset,
            save_path='results/dataset_stats.png'
        )
        print("   ✅ Dataset statistics saved to results/dataset_stats.png")
    except Exception as e:
        print(f"   ❌ Error in dataset analysis: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Visualization analysis completed!")
    print("\n📁 Generated files:")
    results_files = [
        "results/attention_analysis.png",
        "results/quality_analysis.png", 
        "results/threshold_analysis.png",
        "results/feature_maps_layer4.png",
        "results/dataset_stats.png"
    ]
    
    for file_path in results_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (failed to generate)")
    
    print(f"\n🎯 Key Recommendations:")
    print(f"   • Use threshold: {optimal_threshold:.3f} for binary predictions")
    print(f"   • Check attention maps for proper nodule focus")
    print(f"   • Review quality analysis for model performance")
    
    print("\n💡 Next steps:")
    print("   • Open the PNG files to review visualizations")
    print("   • If attention maps look poor, consider retraining") 
    print("   • If Dice scores are low, adjust loss function weights")
    print("   • Use the optimal threshold in your evaluation scripts")

def quick_attention_check():
    """Quick function to just check attention mechanism."""
    
    print("🔍 Quick Attention Check...")
    
    # Minimal setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    try:
        pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = ImprovedXrayDRRSegmentationModel(pretrained_model)
        
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load single sample
    try:
        dataset = DRRSegmentationDataset(
            root_dir='../../../DRR dataset/LIDC_LDRI',
            image_size=(512, 512)
        )
        sample = dataset[0]
        
        xray = sample["xray"].unsqueeze(0).to(device)
        drr = sample["drr"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        
        print("✅ Sample loaded")
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        return
    
    # Quick visualization
    visualizer = ModelVisualizer(model, device)
    visualizer.visualize_attention_mechanism(
        xray=xray, drr=drr, mask=mask,
        save_path='quick_attention_check.png'
    )
    
    print("✅ Quick attention check saved to quick_attention_check.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_attention_check()
    else:
        main()
