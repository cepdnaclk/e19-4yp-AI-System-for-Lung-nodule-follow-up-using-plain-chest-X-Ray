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
        
        # Load model
        self.model = self._load_model(model_path)
        
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
        """Load the trained model."""
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
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def evaluate_model(self, save_results=True):
        """Comprehensive model evaluation."""
        logger.info("Starting model evaluation...")
        
        all_metrics = []
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
                    
                    # Calculate metrics
                    metrics = calculate_metrics(pred_mask, mask, threshold=self.config.INFERENCE_THRESHOLD)
                    metrics['filename'] = filename
                    all_metrics.append(metrics)
                    
                    # Store some samples for visualization
                    if len(sample_predictions) < 20:
                        sample_predictions.append({
                            'xray': xray.cpu(),
                            'pred': pred_mask.cpu(),
                            'mask': mask.cpu(),
                            'filename': filename,
                            'metrics': metrics
                        })
                    
                except Exception as e:
                    logger.error(f"Error evaluating sample {idx}: {e}")
                    continue
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Print results
        self._print_results(aggregated_metrics)
        
        if save_results:
            self._save_results(aggregated_metrics, sample_predictions)
        
        return aggregated_metrics, sample_predictions
    
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
    
    def _save_results(self, metrics, sample_predictions):
        """Save evaluation results."""
        results_dir = self.config.RESULTS_DIR
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics as text
        with open(os.path.join(results_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            if metrics:
                f.write(f"Number of samples: {metrics['num_samples']}\n\n")
                
                metric_names = ['dice', 'iou', 'precision', 'recall', 'f1']
                for metric in metric_names:
                    f.write(f"{metric.upper()}:\n")
                    for stat in ['mean', 'std', 'median', 'min', 'max']:
                        key = f'{metric}_{stat}'
                        if key in metrics:
                            f.write(f"  {stat}: {metrics[key]:.4f}\n")
                    f.write("\n")
        
        # Save sample predictions
        if sample_predictions:
            self._save_sample_visualizations(sample_predictions, results_dir)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _save_sample_visualizations(self, sample_predictions, results_dir):
        """Save visualizations of sample predictions."""
        # Create grid visualization
        num_samples = min(16, len(sample_predictions))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 3, cols, figsize=(20, 15))
        
        for i in range(num_samples):
            sample = sample_predictions[i]
            col = i % cols
            
            # X-ray
            row_xray = (i // cols) * 3
            axes[row_xray, col].imshow(sample['xray'][0, 0].numpy(), cmap='gray')
            axes[row_xray, col].set_title(f"X-ray\n{sample['filename']}", fontsize=8)
            axes[row_xray, col].axis('off')
            
            # Prediction
            row_pred = row_xray + 1
            axes[row_pred, col].imshow(sample['pred'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            dice_score = sample['metrics']['dice']
            axes[row_pred, col].set_title(f"Prediction\nDice: {dice_score:.3f}", fontsize=8)
            axes[row_pred, col].axis('off')
            
            # Ground truth
            row_gt = row_xray + 2
            axes[row_gt, col].imshow(sample['mask'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
            iou_score = sample['metrics']['iou']
            axes[row_gt, col].set_title(f"Ground Truth\nIoU: {iou_score:.3f}", fontsize=8)
            axes[row_gt, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            for j in range(3):
                row = (i // cols) * 3 + j
                col = i % cols
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sample_predictions.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual high-quality samples
        best_samples = sorted(sample_predictions, key=lambda x: x['metrics']['dice'], reverse=True)[:5]
        worst_samples = sorted(sample_predictions, key=lambda x: x['metrics']['dice'])[:5]
        
        for i, sample in enumerate(best_samples):
            self._save_individual_sample(sample, results_dir, f'best_sample_{i+1}')
        
        for i, sample in enumerate(worst_samples):
            self._save_individual_sample(sample, results_dir, f'worst_sample_{i+1}')
    
    def _save_individual_sample(self, sample, results_dir, filename):
        """Save individual sample visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # X-ray
        axes[0].imshow(sample['xray'][0, 0].numpy(), cmap='gray')
        axes[0].set_title('X-ray Input')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(sample['pred'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f"Prediction (Dice: {sample['metrics']['dice']:.3f})")
        axes[1].axis('off')
        
        # Ground truth
        axes[2].imshow(sample['mask'][0, 0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f"Ground Truth (IoU: {sample['metrics']['iou']:.3f})")
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
