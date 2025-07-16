"""
Model Comparison Script for Small Datasets
Compares original vs lightweight architectures on the 545-sample dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

from custom_model import XrayDRRSegmentationModel
from lightweight_model import SmallDatasetXrayDRRModel
from improved_losses import CombinedMedicalLoss
from lightweight_losses import SmallDatasetLoss
from drr_dataset_loading import DRRDataset
from util import dice_coefficient, jaccard_index
from small_dataset_config import SmallDatasetConfig


class ModelComparator:
    """Compare different model architectures on small datasets."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.results = {}
        
    def count_parameters(self, model):
        """Count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def quick_inference_test(self, model, data_loader, model_name, loss_fn):
        """Quick inference test on validation data."""
        model.eval()
        
        losses = []
        dice_scores = []
        jaccard_scores = []
        precisions = []
        recalls = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f'Testing {model_name}')
            for xray, drr, mask in pbar:
                xray = xray.to(self.device)
                drr = drr.to(self.device)
                mask = mask.to(self.device)
                
                # Forward pass
                try:
                    outputs = model(xray, drr)
                    loss = loss_fn(outputs, mask)
                    
                    # Get predictions
                    pred_mask = torch.sigmoid(outputs) > 0.5
                    
                    # Calculate metrics
                    dice = dice_coefficient(pred_mask, mask)
                    jaccard = jaccard_index(pred_mask, mask)
                    
                    # Calculate precision and recall
                    tp = (pred_mask & mask).sum().float()
                    fp = (pred_mask & ~mask).sum().float()
                    fn = (~pred_mask & mask).sum().float()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    
                    losses.append(loss.item())
                    dice_scores.append(dice.item())
                    jaccard_scores.append(jaccard.item())
                    precisions.append(precision.item())
                    recalls.append(recall.item())
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': f'{dice.item():.4f}',
                        'Precision': f'{precision.item():.4f}'
                    })
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
        
        return {
            'loss': np.mean(losses),
            'dice': np.mean(dice_scores),
            'jaccard': np.mean(jaccard_scores),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'loss_std': np.std(losses),
            'dice_std': np.std(dice_scores)
        }
    
    def compare_models(self):
        """Compare original vs lightweight models."""
        print("Setting up comparison...")
        
        # Prepare validation dataset
        val_dataset = DRRDataset(
            data_root=self.config.DATA_ROOT,
            image_size=self.config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=self.config.NORMALIZE_DATA
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,  # Small batch for safety
            shuffle=False,
            num_workers=1
        )
        
        print(f"Validation samples: {len(val_dataset)}")
        
        # Initialize models
        print("\n1. Testing Original Enhanced Model...")
        try:
            original_model = XrayDRRSegmentationModel(
                alpha=0.3,
                pretrained_weights="resnet50-res512-all"
            ).to(self.device)
            
            original_loss = CombinedMedicalLoss(
                pos_weight=100.0,
                alpha=0.7,
                gamma=2.0
            ).to(self.device)
            
            total_params, trainable_params = self.count_parameters(original_model)
            print(f"Original model - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            original_results = self.quick_inference_test(
                original_model, val_loader, "Original Enhanced", original_loss
            )
            self.results['original'] = {
                'metrics': original_results,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
        except Exception as e:
            print(f"Error with original model: {e}")
            self.results['original'] = None
        
        # Test lightweight model
        print("\n2. Testing Lightweight Model...")
        try:
            lightweight_model = SmallDatasetXrayDRRModel(
                alpha=0.2,
                freeze_early_layers=True
            ).to(self.device)
            
            lightweight_loss = SmallDatasetLoss(
                pos_weight=50.0,
                dice_weight=0.6,
                bce_weight=0.4
            ).to(self.device)
            
            total_params, trainable_params = self.count_parameters(lightweight_model)
            print(f"Lightweight model - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            lightweight_results = self.quick_inference_test(
                lightweight_model, val_loader, "Lightweight", lightweight_loss
            )
            self.results['lightweight'] = {
                'metrics': lightweight_results,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
        except Exception as e:
            print(f"Error with lightweight model: {e}")
            self.results['lightweight'] = None
        
        # Test minimal baseline
        print("\n3. Testing Minimal Baseline...")
        try:
            # Create an even simpler baseline
            class MinimalModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Very simple U-Net style
                    self.encoder = nn.Sequential(
                        nn.Conv2d(2, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                    )
                    
                    self.decoder = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, 1)
                    )
                
                def forward(self, xray, drr):
                    x = torch.cat([xray, drr], dim=1)
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x
            
            minimal_model = MinimalModel().to(self.device)
            minimal_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0).to(self.device))
            
            total_params, trainable_params = self.count_parameters(minimal_model)
            print(f"Minimal model - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            minimal_results = self.quick_inference_test(
                minimal_model, val_loader, "Minimal Baseline", minimal_loss
            )
            self.results['minimal'] = {
                'metrics': minimal_results,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
        except Exception as e:
            print(f"Error with minimal model: {e}")
            self.results['minimal'] = None
        
        return self.results
    
    def print_comparison_table(self):
        """Print comparison results in a table format."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        if not self.results:
            print("No results to display.")
            return
        
        # Header
        print(f"{'Model':<15} {'Parameters':<12} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'Loss':<8}")
        print("-" * 80)
        
        for model_name, result in self.results.items():
            if result is None:
                print(f"{model_name:<15} {'ERROR':<12} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
                continue
            
            params = f"{result['trainable_params']:,}"
            metrics = result['metrics']
            
            print(f"{model_name.title():<15} {params:<12} "
                  f"{metrics['dice']:.4f}   {metrics['precision']:.4f}    "
                  f"{metrics['recall']:.4f}   {metrics['loss']:.4f}")
        
        print("-" * 80)
        
        # Print detailed analysis
        print("\nDETAILED ANALYSIS:")
        for model_name, result in self.results.items():
            if result is None:
                continue
            
            metrics = result['metrics']
            print(f"\n{model_name.title()} Model:")
            print(f"  Parameters: {result['trainable_params']:,}")
            print(f"  Dice Score: {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Jaccard Index: {metrics['jaccard']:.4f}")
            print(f"  Loss: {metrics['loss']:.4f} ± {metrics['loss_std']:.4f}")
    
    def plot_comparison(self, save_dir):
        """Create comparison plots."""
        if not self.results:
            return
        
        # Extract data for plotting
        models = []
        dice_scores = []
        precisions = []
        recalls = []
        parameters = []
        
        for model_name, result in self.results.items():
            if result is None:
                continue
            
            models.append(model_name.title())
            dice_scores.append(result['metrics']['dice'])
            precisions.append(result['metrics']['precision'])
            recalls.append(result['metrics']['recall'])
            parameters.append(result['trainable_params'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dice scores
        bars1 = ax1.bar(models, dice_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_ylabel('Dice Score')
        ax1.set_title('Dice Score Comparison')
        ax1.set_ylim(0, max(dice_scores) * 1.2)
        for bar, score in zip(bars1, dice_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # Precision vs Recall
        ax2.scatter(recalls, precisions, s=[p/1000 for p in parameters], 
                   c=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs Recall (bubble size = parameters)')
        
        # Parameters comparison
        bars3 = ax3.bar(models, [p/1000 for p in parameters], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_ylabel('Parameters (thousands)')
        ax3.set_title('Model Complexity (Parameters)')
        for bar, param in zip(bars3, parameters):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{param:,}', ha='center', va='bottom', rotation=45)
        
        # Combined metric (F1-score)
        f1_scores = [2 * (p * r) / (p + r + 1e-8) for p, r in zip(precisions, recalls)]
        bars4 = ax4.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score Comparison')
        for bar, f1 in zip(bars4, f1_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{f1:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to {save_dir}/model_comparison.png")
    
    def save_results(self, save_dir):
        """Save comparison results to JSON."""
        # Prepare data for JSON serialization
        json_results = {}
        for model_name, result in self.results.items():
            if result is None:
                json_results[model_name] = None
            else:
                json_results[model_name] = {
                    'metrics': {k: float(v) for k, v in result['metrics'].items()},
                    'total_params': int(result['total_params']),
                    'trainable_params': int(result['trainable_params'])
                }
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(DRRDataset(
                data_root=self.config.DATA_ROOT,
                image_size=self.config.IMAGE_SIZE,
                training=False,
                augment=False
            )),
            'results': json_results,
            'config': str(self.config)
        }
        
        with open(os.path.join(save_dir, 'model_comparison_results.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Results saved to {save_dir}/model_comparison_results.json")


def main():
    """Main comparison function."""
    config = SmallDatasetConfig()
    config.create_directories()
    
    print("Starting Model Comparison for Small Dataset...")
    print(f"Dataset: {config.DATA_ROOT}")
    print(f"Device: {config.DEVICE}")
    
    comparator = ModelComparator(config)
    results = comparator.compare_models()
    
    # Print results
    comparator.print_comparison_table()
    
    # Create visualizations
    comparator.plot_comparison(config.LOG_DIR)
    
    # Save results
    comparator.save_results(config.LOG_DIR)
    
    print("\nComparison completed!")
    print(f"Results saved to: {config.LOG_DIR}")
    
    return results


if __name__ == "__main__":
    results = main()
