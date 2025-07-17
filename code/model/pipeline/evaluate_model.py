"""
Standalone evaluation script for trained improved models.
Generates comprehensive evaluation reports and visualizations.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path

from improved_model import create_improved_model
from improved_config import IMPROVED_CONFIG
from drr_dataset_loading import DRRDataset
from training_visualization import EvaluationVisualizer, create_final_visualization_report

def find_dataset_path():
    """Find the dataset path using the same logic as training script."""
    data_root = "../../DRR dataset/LIDC_LDRI"
    
    if not os.path.exists(data_root):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        data_root = os.path.join(project_root, "DRR dataset", "LIDC_LDRI")
        
        if not os.path.exists(data_root):
            for root, dirs, files in os.walk(os.path.dirname(project_root)):
                if "LIDC_LDRI" in dirs:
                    data_root = os.path.join(root, "LIDC_LDRI")
                    break
    
    return data_root

def evaluate_model(model_path, dataset_split='validation', batch_size=None, save_dir=None):
    """
    Evaluate a trained model comprehensively.
    
    Args:
        model_path: Path to the model checkpoint
        dataset_split: 'validation' or 'training'
        batch_size: Batch size for evaluation (uses config default if None)
        save_dir: Directory to save evaluation results
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = batch_size or IMPROVED_CONFIG['BATCH_SIZE']
    save_dir = save_dir or f'evaluation_results_{dataset_split}'
    
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Dataset Split: {dataset_split}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Save Directory: {save_dir}")
    print("-"*60)
    
    # Create evaluation visualizer
    eval_visualizer = EvaluationVisualizer(save_dir=save_dir)
    
    # Load dataset
    data_root = find_dataset_path()
    print(f"Dataset Path: {data_root}")
    
    dataset = DRRDataset(
        data_root=data_root,
        training=(dataset_split == 'training'),
        augment=False,  # No augmentation for evaluation
        normalize=IMPROVED_CONFIG['NORMALIZE_IMAGES']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Load model
    print("Loading model...")
    model = create_improved_model()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {model_path}")
        
        # Print model info if available
        if 'epoch' in checkpoint:
            print(f"  - Trained for {checkpoint['epoch']+1} epochs")
        if 'best_dice' in checkpoint:
            print(f"  - Best training dice: {checkpoint['best_dice']:.4f}")
    else:
        print(f"⚠️ Model checkpoint not found: {model_path}")
        return None
    
    model = model.to(device)
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    eval_report = eval_visualizer.evaluate_model_detailed(model, dataloader, device)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    summary = eval_report['evaluation_summary']
    categories = eval_report['performance_categories']
    
    print(f"Total Samples Evaluated: {summary['total_samples']}")
    print(f"Mean Dice Score: {summary['mean_dice_score']:.4f} ± {summary['std_dice_score']:.4f}")
    print(f"Median Dice Score: {summary['median_dice_score']:.4f}")
    print(f"Range: [{summary['min_dice_score']:.4f}, {summary['max_dice_score']:.4f}]")
    print(f"Mean Loss: {summary['mean_loss']:.4f}")
    print()
    print("Performance Breakdown:")
    print(f"  Excellent (≥0.8): {categories['excellent_samples']} samples ({categories['excellent_percentage']:.1f}%)")
    print(f"  Good (0.6-0.8): {categories['good_samples']} samples ({categories['good_percentage']:.1f}%)")
    print(f"  Fair (0.4-0.6): {categories['fair_samples']} samples ({categories['fair_percentage']:.1f}%)")
    print(f"  Poor (<0.4): {categories['poor_samples']} samples ({categories['poor_percentage']:.1f}%)")
    
    print(f"\n✓ Evaluation completed! Results saved to: {save_dir}/")
    print(f"✓ Check {save_dir}/evaluation_summary.txt for detailed report")
    
    return eval_report

def compare_models(model_paths, model_names=None, save_dir='model_comparison'):
    """Compare multiple trained models."""
    
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\nEvaluating {model_name}...")
        
        if os.path.exists(model_path):
            eval_dir = f"{save_dir}/{model_name}_evaluation"
            result = evaluate_model(model_path, save_dir=eval_dir)
            if result:
                results[model_name] = result
        else:
            print(f"⚠️ Model not found: {model_path}")
    
    if len(results) > 1:
        # Create comparison report
        comparison_dir = Path(save_dir)
        comparison_dir.mkdir(exist_ok=True)
        
        print(f"\n{'Model':<15} {'Mean Dice':<12} {'Std Dice':<12} {'Excellent %':<12}")
        print("-" * 60)
        
        for model_name, result in results.items():
            summary = result['evaluation_summary']
            categories = result['performance_categories']
            
            print(f"{model_name:<15} {summary['mean_dice_score']:<12.4f} "
                  f"{summary['std_dice_score']:<12.4f} {categories['excellent_percentage']:<12.1f}")
        
        print(f"\n✓ Model comparison completed! Results saved to: {save_dir}/")
    
    return results

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained improved models')
    parser.add_argument('--model', type=str, 
                       default='checkpoints_improved/best_model_improved.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, choices=['training', 'validation'], 
                       default='validation',
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for evaluation')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                       help='Paths to multiple models for comparison')
    parser.add_argument('--compare-names', nargs='+', type=str, default=None,
                       help='Names for models in comparison')
    parser.add_argument('--final-report', action='store_true',
                       help='Generate final comprehensive report')
    
    args = parser.parse_args()
    
    if args.compare:
        # Model comparison mode
        model_names = args.compare_names or [f"Model_{i+1}" for i in range(len(args.compare))]
        compare_models(args.compare, model_names, 'model_comparison')
    else:
        # Single model evaluation
        evaluate_model(
            model_path=args.model,
            dataset_split=args.split,
            batch_size=args.batch_size,
            save_dir=args.save_dir
        )
    
    if args.final_report:
        print("\nGenerating final comprehensive report...")
        try:
            create_final_visualization_report()
            print("✓ Final report generated in final_report/")
        except Exception as e:
            print(f"⚠️ Error generating final report: {e}")

if __name__ == "__main__":
    main()
