"""
Evaluation script for the DRR segmentation model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm

from drr_dataset_loading import DRRSegmentationDataset
from custom_model import XrayDRRSegmentationModel
from util import calculate_metrics, show_prediction_vs_groundtruth
from config import Config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model_path = model_path
        
        # Load model and threshold info
        self.model, self.threshold_info = self._load_model(model_path)
        
        # Setup dataset
        self.test_dataset = DRRSegmentationDataset(
            root_dir=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            augment=False,  # No augmentation for evaluation
            normalize=self.config.NORMALIZE_DATA
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,  # Evaluate one by one for detailed analysis
            shuffle=False,
            num_workers=2
        )
        
    def _load_model(self, model_path):
        """Load the trained model and threshold info."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load pretrained backbone
            import torchxrayvision as xrv
            pretrained_model = xrv.models.ResNet(weights=self.config.PRETRAINED_WEIGHTS)
            
            # Initialize model
            model = XrayDRRSegmentationModel(
                pretrained_model, 
                alpha=checkpoint.get('config', {}).get('alpha', self.config.ALPHA)
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Load threshold info
            threshold_info = checkpoint.get('threshold_info', {
                'optimal_threshold': self.config.INFERENCE_THRESHOLD,
                'best_dice': 0.0,
                'default_threshold': 0.5
            })
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
            logger.info(f"Optimal threshold: {threshold_info['optimal_threshold']:.4f}")
            
            return model, threshold_info
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def evaluate_model(self, save_results=True):
        """Comprehensive model evaluation."""
        logger.info("Starting model evaluation...")
        
        optimal_threshold = self.threshold_info['optimal_threshold']
        logger.info(f"Using optimal threshold: {optimal_threshold:.4f}")
        
        all_metrics_default = []
        all_metrics_optimal = []
        sample_predictions = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                try:
                    # Move to device
                    xray = batch["xray"].to(self.device)
                    drr = batch["drr"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    filename = batch["filename"][0]
                    
                    # Forward pass
                    pred_mask = self.model(xray, drr)
                    
                    # Calculate metrics with default threshold
                    metrics_default = calculate_metrics(pred_mask, mask, threshold=0.5)
                    metrics_default['filename'] = filename
                    all_metrics_default.append(metrics_default)
                    
                    # Calculate metrics with optimal threshold
                    metrics_optimal = calculate_metrics(pred_mask, mask, threshold=optimal_threshold)
                    metrics_optimal['filename'] = filename
                    all_metrics_optimal.append(metrics_optimal)
                    
                    # Store some samples for visualization
                    if len(sample_predictions) < 20:
                        sample_predictions.append({
                            'xray': xray.cpu(),
                            'pred': pred_mask.cpu(),
                            'pred_optimal': (pred_mask > optimal_threshold).float().cpu(),
                            'mask': mask.cpu(),
                            'filename': filename,
                            'metrics_default': metrics_default,
                            'metrics_optimal': metrics_optimal
                        })
                    
                except Exception as e:
                    logger.error(f"Error evaluating sample {idx}: {e}")
                    continue
        
        # Aggregate metrics
        aggregated_metrics_default = self._aggregate_metrics(all_metrics_default)
        aggregated_metrics_optimal = self._aggregate_metrics(all_metrics_optimal)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS WITH DEFAULT THRESHOLD (0.5)")
        print("="*60)
        self._print_results(aggregated_metrics_default)
        
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f})")
        print("="*60)
        self._print_results(aggregated_metrics_optimal)
        
        if save_results:
            self._save_results(aggregated_metrics_default, aggregated_metrics_optimal, sample_predictions)
        
        return aggregated_metrics_default, aggregated_metrics_optimal, sample_predictions
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across all samples."""
        if not all_metrics:
            return {}
        
        metric_keys = ['dice', 'iou', 'precision', 'recall', 'f1']
        aggregated = {}
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_median'] = np.median(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        aggregated['num_samples'] = len(all_metrics)
        
        return aggregated
    
    def _print_results(self, metrics):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if not metrics:
            print("No valid metrics calculated!")
            return
        
        print(f"Number of samples evaluated: {metrics['num_samples']}")
        print()
        
        metric_names = ['dice', 'iou', 'precision', 'recall', 'f1']
        for metric in metric_names:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in metrics and std_key in metrics:
                print(f"{metric.upper():>10}: {metrics[mean_key]:.4f} ± {metrics[std_key]:.4f}")
        
        print("\nDetailed Statistics:")
        print("-" * 40)
        for metric in metric_names:
            print(f"\n{metric.upper()}:")
            for stat in ['mean', 'std', 'median', 'min', 'max']:
                key = f'{metric}_{stat}'
                if key in metrics:
                    print(f"  {stat:>8}: {metrics[key]:.4f}")
        
        print("="*60)
    
    def _save_results(self, metrics_default, metrics_optimal, sample_predictions):
        """Save evaluation results."""
        results_dir = self.config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as text
        with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            # Default threshold results
            f.write("DEFAULT THRESHOLD (0.5) RESULTS:\n")
            f.write("-" * 40 + "\n")
            if metrics_default:
                f.write(f"Number of samples: {metrics_default['num_samples']}\n\n")
                
                metric_names = ['dice', 'iou', 'precision', 'recall', 'f1']
                for metric in metric_names:
                    f.write(f"{metric.upper()}:\n")
                    for stat in ['mean', 'std', 'median', 'min', 'max']:
                        key = f'{metric}_{stat}'
                        if key in metrics_default:
                            f.write(f"  {stat}: {metrics_default[key]:.4f}\n")
                    f.write("\n")
            
            # Optimal threshold results
            f.write(f"\nOPTIMAL THRESHOLD ({self.threshold_info['optimal_threshold']:.4f}) RESULTS:\n")
            f.write("-" * 40 + "\n")
            if metrics_optimal:
                f.write(f"Number of samples: {metrics_optimal['num_samples']}\n\n")
                
                metric_names = ['dice', 'iou', 'precision', 'recall', 'f1']
                for metric in metric_names:
                    f.write(f"{metric.upper()}:\n")
                    for stat in ['mean', 'std', 'median', 'min', 'max']:
                        key = f'{metric}_{stat}'
                        if key in metrics_optimal:
                            f.write(f"  {stat}: {metrics_optimal[key]:.4f}\n")
                    f.write("\n")
        
        # Save sample predictions
        if sample_predictions:
            self._save_sample_visualizations(sample_predictions, results_dir)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _save_sample_visualizations(self, sample_predictions, results_dir):
        """Save visualizations of sample predictions."""
        # Create grid visualization for default threshold
        num_samples = min(12, len(sample_predictions))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        # Default threshold visualization
        fig, axes = plt.subplots(rows * 3, cols, figsize=(20, 15))
        
        for i in range(num_samples):
            sample = sample_predictions[i]
            col = i % cols
            
            # X-ray
            row_xray = (i // cols) * 3
            axes[row_xray, col].imshow(sample['xray'][0, 0].numpy(), cmap='gray')
            axes[row_xray, col].set_title(f"X-ray\n{sample['filename']}", fontsize=8)
            axes[row_xray, col].axis('off')
            
            # Prediction (default threshold)
            row_pred = row_xray + 1
            axes[row_pred, col].imshow(sample['pred'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            dice_score = sample['metrics_default']['dice']
            axes[row_pred, col].set_title(f"Pred (0.5)\nDice: {dice_score:.3f}", fontsize=8)
            axes[row_pred, col].axis('off')
            
            # Ground truth
            row_gt = row_xray + 2
            axes[row_gt, col].imshow(sample['mask'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            iou_score = sample['metrics_default']['iou']
            axes[row_gt, col].set_title(f"Ground Truth\nIoU: {iou_score:.3f}", fontsize=8)
            axes[row_gt, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            for j in range(3):
                row = (i // cols) * 3 + j
                col = i % cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sample_predictions_default.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Optimal threshold visualization
        fig, axes = plt.subplots(rows * 3, cols, figsize=(20, 15))
        
        for i in range(num_samples):
            sample = sample_predictions[i]
            col = i % cols
            
            # X-ray
            row_xray = (i // cols) * 3
            axes[row_xray, col].imshow(sample['xray'][0, 0].numpy(), cmap='gray')
            axes[row_xray, col].set_title(f"X-ray\n{sample['filename']}", fontsize=8)
            axes[row_xray, col].axis('off')
            
            # Prediction (optimal threshold)
            row_pred = row_xray + 1
            axes[row_pred, col].imshow(sample['pred_optimal'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            dice_score = sample['metrics_optimal']['dice']
            optimal_th = self.threshold_info['optimal_threshold']
            axes[row_pred, col].set_title(f"Pred ({optimal_th:.2f})\nDice: {dice_score:.3f}", fontsize=8)
            axes[row_pred, col].axis('off')
            
            # Ground truth
            row_gt = row_xray + 2
            axes[row_gt, col].imshow(sample['mask'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            iou_score = sample['metrics_optimal']['iou']
            axes[row_gt, col].set_title(f"Ground Truth\nIoU: {iou_score:.3f}", fontsize=8)
            axes[row_gt, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            for j in range(3):
                row = (i // cols) * 3 + j
                col = i % cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sample_predictions_optimal.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual high-quality samples
        best_samples_default = sorted(sample_predictions, key=lambda x: x['metrics_default']['dice'], reverse=True)[:5]
        best_samples_optimal = sorted(sample_predictions, key=lambda x: x['metrics_optimal']['dice'], reverse=True)[:5]
        
        for i, sample in enumerate(best_samples_default):
            self._save_individual_sample(sample, results_dir, f'best_default_{i+1}', 'metrics_default', 'pred')
        
        for i, sample in enumerate(best_samples_optimal):
            self._save_individual_sample(sample, results_dir, f'best_optimal_{i+1}', 'metrics_optimal', 'pred_optimal')
    
    def _save_individual_sample(self, sample, results_dir, filename, metrics_key, pred_key):
        """Save individual sample visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # X-ray
        axes[0].imshow(sample['xray'][0, 0].numpy(), cmap='gray')
        axes[0].set_title('X-ray Input')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(sample[pred_key][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Prediction (Dice: {sample[metrics_key]['dice']:.3f})")
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(sample['mask'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f"Ground Truth (IoU: {sample[metrics_key]['iou']:.3f})")
        axes[2].axis('off')
        
        plt.suptitle(f"Sample: {sample['filename']}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{filename}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DRR Segmentation Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--config', type=str, choices=['dev', 'prod'], default='prod',
                       help='Configuration to use')
    
    args = parser.parse_args()
    
    # Setup configuration
    if args.config == 'dev':
        from config import DevConfig as Config
    else:
        from config import ProdConfig as Config
    
    Config.create_directories()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, Config)
    metrics, predictions = evaluator.evaluate_model(save_results=True)

if __name__ == "__main__":
    main()
