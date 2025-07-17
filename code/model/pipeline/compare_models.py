"""
Comprehensive Model Comparison Script
Compares the improved model with the baseline pretrained model.
Provides detailed metrics, visualizations, and statistical analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import stats
import argparse

# Import your models and utilities
from improved_model import ImprovedXrayDRRModel
from improved_config import IMPROVED_CONFIG
from drr_dataset_loading import DRRDataset
from training_visualization import EvaluationVisualizer

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BaselineModel(nn.Module):
    """Baseline model using only pretrained ResNet for comparison."""
    
    def __init__(self, pretrained_model=None):
        super().__init__()
        
        if pretrained_model is None:
            pretrained_model = xrv.models.ResNet(weights="resnet50-res512-all")
        
        self.pretrained_model = pretrained_model
        
        # Feature extractor (same as improved model)
        self.feature_extractor = nn.Sequential(
            pretrained_model.model.conv1,
            pretrained_model.model.bn1,
            pretrained_model.model.relu,
            pretrained_model.model.maxpool,
            pretrained_model.model.layer1,
            pretrained_model.model.layer2,
            pretrained_model.model.layer3,
            pretrained_model.model.layer4,
        )
        
        # Simple segmentation head without attention
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Progressive upsampling
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16->32
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32->64
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64->128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128->256
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 256->512
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
    def forward(self, xray, drr=None):
        """Forward pass - only uses X-ray input."""
        # Extract features from X-ray only
        features = self.feature_extractor(xray)  # [B, 2048, 16, 16]
        
        # Generate segmentation
        segmentation = self.segmentation_head(features)
        
        return {
            'segmentation': segmentation,
            'attention': None,  # No attention mechanism
            'nodule_features': None
        }

class ModelComparator:
    """Comprehensive model comparison framework."""
    
    def __init__(self, improved_model_path, save_dir='model_comparison'):
        self.improved_model_path = improved_model_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.improved_model = None
        self.baseline_model = None
        
        # Results storage
        self.comparison_results = {}
        
    def load_models(self):
        """Load both improved and baseline models."""
        print("Loading models...")
        
        # Load improved model
        self.improved_model = ImprovedXrayDRRModel(
            alpha=IMPROVED_CONFIG.get('ALPHA', 0.3),
            use_supervised_attention=IMPROVED_CONFIG.get('USE_SUPERVISED_ATTENTION', True),
            target_pathology=IMPROVED_CONFIG.get('TARGET_PATHOLOGY', 'Nodule')
        )
        
        if os.path.exists(self.improved_model_path):
            checkpoint = torch.load(self.improved_model_path, map_location=self.device)
            self.improved_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Improved model loaded from {self.improved_model_path}")
            
            if 'epoch' in checkpoint:
                print(f"  - Trained for {checkpoint['epoch']+1} epochs")
            if 'best_dice' in checkpoint:
                print(f"  - Best training dice: {checkpoint['best_dice']:.4f}")
        else:
            raise FileNotFoundError(f"Improved model not found: {self.improved_model_path}")
        
        # Load baseline model (pretrained only)
        self.baseline_model = BaselineModel()
        print("✓ Baseline model initialized with pretrained weights")
        
        # Move models to device
        self.improved_model = self.improved_model.to(self.device)
        self.baseline_model = self.baseline_model.to(self.device)
        
    def load_dataset(self, split='validation'):
        """Load dataset for evaluation."""
        print(f"Loading {split} dataset...")
        
        # Find dataset path
        data_root = self._find_dataset_path()
        
        dataset = DRRDataset(
            data_root=data_root,
            training=(split == 'training'),
            augment=False,
            normalize=IMPROVED_CONFIG['NORMALIZE_IMAGES']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=IMPROVED_CONFIG['BATCH_SIZE'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataloader
    
    def _find_dataset_path(self):
        """Find dataset path."""
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
    
    def evaluate_models(self, dataloader):
        """Evaluate both models on the dataset."""
        print("\\nEvaluating models...")
        
        models = {
            'Improved Model': self.improved_model,
            'Baseline Model': self.baseline_model
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\\nEvaluating {model_name}...")
            model.eval()
            
            all_predictions = []
            all_ground_truth = []
            all_dice_scores = []
            all_losses = []
            sample_results = []
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    try:
                        if isinstance(batch, dict):
                            xray = batch["xray"].to(self.device)
                            drr = batch["drr"].to(self.device)
                            masks = batch["mask"].to(self.device)
                        else:
                            if len(batch) >= 3:
                                xray, drr, masks = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                            else:
                                continue
                        
                        # Forward pass
                        if model_name == 'Improved Model':
                            outputs = model(xray, drr)
                        else:
                            outputs = model(xray)  # Baseline only uses X-ray
                        
                        # Get predictions
                        if isinstance(outputs, dict):
                            segmentation_output = outputs['segmentation']
                        else:
                            segmentation_output = outputs
                        
                        predictions = torch.sigmoid(segmentation_output)
                        
                        # Calculate loss
                        loss = F.binary_cross_entropy_with_logits(segmentation_output, masks)
                        all_losses.append(loss.item())
                        
                        # Calculate metrics for each sample
                        for j in range(predictions.shape[0]):
                            if torch.is_tensor(predictions):
                                pred = predictions[j].cpu().numpy()
                            else:
                                pred = predictions[j]
                            
                            if torch.is_tensor(masks):
                                mask = masks[j].cpu().numpy()
                            else:
                                mask = masks[j]
                            
                            # Binary prediction
                            pred_binary = (pred > 0.5).astype(float)
                            
                            # Calculate dice
                            intersection = (pred_binary * mask).sum()
                            dice = (2 * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)
                            all_dice_scores.append(dice)
                            
                            # Store for analysis
                            all_predictions.append(pred.flatten())
                            all_ground_truth.append(mask.flatten())
                            
                            # Store sample for visualization
                            if len(sample_results) < 8:
                                sample_results.append({
                                    'xray': xray[j].cpu().numpy() if torch.is_tensor(xray) else xray[j],
                                    'drr': drr[j].cpu().numpy() if torch.is_tensor(drr) else drr[j],
                                    'ground_truth': mask,
                                    'prediction': pred,
                                    'dice': dice,
                                    'attention': outputs.get('attention', None)
                                })
                    
                    except Exception as e:
                        print(f"Error processing batch {i}: {e}")
                        continue
            
            # Store results
            results[model_name] = {
                'dice_scores': np.array(all_dice_scores),
                'losses': np.array(all_losses),
                'predictions': all_predictions,
                'ground_truth': all_ground_truth,
                'sample_results': sample_results,
                'summary': self._calculate_summary_stats(all_dice_scores, all_losses)
            }
            
            print(f"✓ {model_name} evaluation completed")
            print(f"  Mean Dice: {results[model_name]['summary']['mean_dice']:.4f}")
            print(f"  Mean Loss: {results[model_name]['summary']['mean_loss']:.4f}")
        
        self.comparison_results = results
        return results
    
    def _calculate_summary_stats(self, dice_scores, losses):
        """Calculate summary statistics."""
        dice_scores = np.array(dice_scores)
        losses = np.array(losses)
        
        return {
            'mean_dice': float(np.mean(dice_scores)),
            'std_dice': float(np.std(dice_scores)),
            'median_dice': float(np.median(dice_scores)),
            'min_dice': float(np.min(dice_scores)),
            'max_dice': float(np.max(dice_scores)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'total_samples': len(dice_scores),
            'excellent_samples': int((dice_scores >= 0.8).sum()),
            'good_samples': int(((dice_scores >= 0.6) & (dice_scores < 0.8)).sum()),
            'fair_samples': int(((dice_scores >= 0.4) & (dice_scores < 0.6)).sum()),
            'poor_samples': int((dice_scores < 0.4).sum())
        }
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations."""
        print("\\nCreating comparison visualizations...")
        
        # 1. Performance comparison plots
        self._plot_performance_comparison()
        
        # 2. Distribution comparison
        self._plot_distribution_comparison()
        
        # 3. Sample predictions comparison
        self._plot_sample_predictions_comparison()
        
        # 4. ROC and PR curves
        self._plot_roc_pr_curves()
        
        # 5. Statistical analysis
        self._plot_statistical_analysis()
        
        print("✓ All visualizations created")
    
    def _plot_performance_comparison(self):
        """Plot side-by-side performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.comparison_results.keys())
        
        # Dice score comparison
        dice_data = [self.comparison_results[model]['dice_scores'] for model in models]
        axes[0, 0].boxplot(dice_data, labels=models)
        axes[0, 0].set_title('Dice Score Distribution')
        axes[0, 0].set_ylabel('Dice Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        loss_data = [self.comparison_results[model]['losses'] for model in models]
        axes[0, 1].boxplot(loss_data, labels=models)
        axes[0, 1].set_title('Loss Distribution')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance categories
        categories = ['Excellent (≥0.8)', 'Good (0.6-0.8)', 'Fair (0.4-0.6)', 'Poor (<0.4)']
        improved_data = [
            self.comparison_results['Improved Model']['summary']['excellent_samples'],
            self.comparison_results['Improved Model']['summary']['good_samples'],
            self.comparison_results['Improved Model']['summary']['fair_samples'],
            self.comparison_results['Improved Model']['summary']['poor_samples']
        ]
        baseline_data = [
            self.comparison_results['Baseline Model']['summary']['excellent_samples'],
            self.comparison_results['Baseline Model']['summary']['good_samples'],
            self.comparison_results['Baseline Model']['summary']['fair_samples'],
            self.comparison_results['Baseline Model']['summary']['poor_samples']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, improved_data, width, label='Improved Model', alpha=0.8)
        axes[1, 0].bar(x + width/2, baseline_data, width, label='Baseline Model', alpha=0.8)
        axes[1, 0].set_title('Performance Category Distribution')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        stats_data = []
        for model in models:
            summary = self.comparison_results[model]['summary']
            stats_data.append([
                f"{summary['mean_dice']:.4f}",
                f"{summary['std_dice']:.4f}",
                f"{summary['median_dice']:.4f}",
                f"{summary['mean_loss']:.4f}"
            ])
        
        table = axes[1, 1].table(
            cellText=stats_data,
            rowLabels=models,
            colLabels=['Mean Dice', 'Std Dice', 'Median Dice', 'Mean Loss'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_comparison(self):
        """Plot distribution comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Distribution Comparison', fontsize=16, fontweight='bold')
        
        # Dice score histograms
        for i, model in enumerate(self.comparison_results.keys()):
            dice_scores = self.comparison_results[model]['dice_scores']
            axes[0, i].hist(dice_scores, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].axvline(np.mean(dice_scores), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(dice_scores):.4f}')
            axes[0, i].axvline(np.median(dice_scores), color='orange', linestyle='--',
                              label=f'Median: {np.median(dice_scores):.4f}')
            axes[0, i].set_title(f'{model} - Dice Score Distribution')
            axes[0, i].set_xlabel('Dice Score')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Cumulative distribution
        for model in self.comparison_results.keys():
            dice_scores = self.comparison_results[model]['dice_scores']
            sorted_dice = np.sort(dice_scores)
            cumulative = np.arange(1, len(sorted_dice) + 1) / len(sorted_dice)
            axes[1, 0].plot(sorted_dice, cumulative, linewidth=2, label=model)
        
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].set_xlabel('Dice Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Violin plot comparison
        dice_data = []
        labels = []
        for model in self.comparison_results.keys():
            dice_data.extend(self.comparison_results[model]['dice_scores'])
            labels.extend([model] * len(self.comparison_results[model]['dice_scores']))
        
        df = pd.DataFrame({'Model': labels, 'Dice Score': dice_data})
        sns.violinplot(data=df, x='Model', y='Dice Score', ax=axes[1, 1])
        axes[1, 1].set_title('Dice Score Distribution (Violin Plot)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_predictions_comparison(self):
        """Plot sample predictions from both models."""
        fig, axes = plt.subplots(5, 8, figsize=(24, 15))
        fig.suptitle('Sample Predictions Comparison', fontsize=16, fontweight='bold')
        
        # Get sample results from improved model (they should have same inputs)
        improved_samples = self.comparison_results['Improved Model']['sample_results'][:8]
        baseline_samples = self.comparison_results['Baseline Model']['sample_results'][:8]
        
        for i in range(min(8, len(improved_samples))):
            # X-ray input
            axes[0, i].imshow(improved_samples[i]['xray'][0], cmap='gray')
            axes[0, i].set_title(f'Sample {i+1}: X-ray')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(improved_samples[i]['ground_truth'][0], cmap='hot')
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Improved model prediction
            axes[2, i].imshow(improved_samples[i]['prediction'][0], cmap='hot')
            axes[2, i].set_title(f'Improved (Dice: {improved_samples[i]["dice"]:.3f})')
            axes[2, i].axis('off')
            
            # Baseline model prediction
            axes[3, i].imshow(baseline_samples[i]['prediction'][0], cmap='hot')
            axes[3, i].set_title(f'Baseline (Dice: {baseline_samples[i]["dice"]:.3f})')
            axes[3, i].axis('off')
            
            # Attention map (if available)
            if improved_samples[i]['attention'] is not None:
                attention = improved_samples[i]['attention']
                if torch.is_tensor(attention):
                    attention = attention.cpu().numpy()
                axes[4, i].imshow(attention[0, 0], cmap='jet', alpha=0.7)
                axes[4, i].set_title('Attention Map')
            else:
                axes[4, i].text(0.5, 0.5, 'No Attention', ha='center', va='center')
                axes[4, i].set_title('No Attention')
            axes[4, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_predictions_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_pr_curves(self):
        """Plot ROC and Precision-Recall curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('ROC and Precision-Recall Curves', fontsize=16, fontweight='bold')
        
        for model_name in self.comparison_results.keys():
            # Flatten predictions and ground truth
            y_true = np.concatenate([gt.flatten() for gt in self.comparison_results[model_name]['ground_truth']])
            y_pred = np.concatenate([pred.flatten() for pred in self.comparison_results[model_name]['predictions']])
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            axes[1].plot(recall, precision, linewidth=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        # ROC plot formatting
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PR plot formatting
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self):
        """Plot statistical significance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
        
        improved_dice = self.comparison_results['Improved Model']['dice_scores']
        baseline_dice = self.comparison_results['Baseline Model']['dice_scores']
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(improved_dice, baseline_dice)
        wilcoxon_stat, wilcoxon_p = stats.mannwhitneyu(improved_dice, baseline_dice, alternative='two-sided')
        
        # Scatter plot comparison
        min_len = min(len(improved_dice), len(baseline_dice))
        axes[0, 0].scatter(improved_dice[:min_len], baseline_dice[:min_len], alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Improved Model Dice')
        axes[0, 0].set_ylabel('Baseline Model Dice')
        axes[0, 0].set_title('Per-Sample Dice Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Difference histogram
        dice_diff = improved_dice[:min_len] - baseline_dice[:min_len]
        axes[0, 1].hist(dice_diff, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].axvline(np.mean(dice_diff), color='green', linestyle='--', 
                          label=f'Mean Diff: {np.mean(dice_diff):.4f}')
        axes[0, 1].set_xlabel('Dice Difference (Improved - Baseline)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Performance Difference Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Statistical test results
        stats_text = f'''
        Statistical Test Results:
        
        T-test:
        • t-statistic: {t_stat:.4f}
        • p-value: {p_value:.6f}
        • Significant: {p_value < 0.05}
        
        Mann-Whitney U test:
        • U-statistic: {wilcoxon_stat:.4f}
        • p-value: {wilcoxon_p:.6f}
        • Significant: {wilcoxon_p < 0.05}
        
        Effect Size:
        • Mean difference: {np.mean(dice_diff):.4f}
        • Cohen's d: {(np.mean(improved_dice) - np.mean(baseline_dice)) / np.sqrt((np.var(improved_dice) + np.var(baseline_dice)) / 2):.4f}
        
        Sample Statistics:
        • Improved samples: {len(improved_dice)}
        • Baseline samples: {len(baseline_dice)}
        • Paired samples: {min_len}
        '''
        
        axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 0].axis('off')
        
        # Performance improvement analysis
        improvement_data = {
            'Metric': ['Mean Dice', 'Median Dice', 'Std Dice', 'Max Dice', 'Mean Loss'],
            'Improved Model': [
                self.comparison_results['Improved Model']['summary']['mean_dice'],
                self.comparison_results['Improved Model']['summary']['median_dice'],
                self.comparison_results['Improved Model']['summary']['std_dice'],
                self.comparison_results['Improved Model']['summary']['max_dice'],
                self.comparison_results['Improved Model']['summary']['mean_loss']
            ],
            'Baseline Model': [
                self.comparison_results['Baseline Model']['summary']['mean_dice'],
                self.comparison_results['Baseline Model']['summary']['median_dice'],
                self.comparison_results['Baseline Model']['summary']['std_dice'],
                self.comparison_results['Baseline Model']['summary']['max_dice'],
                self.comparison_results['Baseline Model']['summary']['mean_loss']
            ]
        }
        
        df_comparison = pd.DataFrame(improvement_data)
        df_comparison['Improvement'] = df_comparison['Improved Model'] - df_comparison['Baseline Model']
        df_comparison['Improvement %'] = (df_comparison['Improvement'] / df_comparison['Baseline Model']) * 100
        
        # Plot improvement bars
        metrics = df_comparison['Metric']
        improvements = df_comparison['Improvement %']
        colors = ['green' if x > 0 else 'red' for x in improvements]
        
        bars = axes[1, 1].bar(metrics, improvements, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Relative Improvement')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                           f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\\nGenerating comparison report...")
        
        # Create detailed comparison report
        report = {
            'comparison_summary': {
                'timestamp': datetime.now().isoformat(),
                'improved_model_path': str(self.improved_model_path),
                'dataset_samples': len(self.comparison_results['Improved Model']['dice_scores']),
                'evaluation_device': str(self.device)
            },
            'model_results': {},
            'statistical_analysis': self._perform_statistical_analysis(),
            'conclusions': self._generate_conclusions()
        }
        
        # Add results for each model
        for model_name in self.comparison_results.keys():
            report['model_results'][model_name] = self.comparison_results[model_name]['summary']
        
        # Save JSON report
        with open(self.save_dir / 'model_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable report
        self._save_readable_comparison_report(report)
        
        print(f"✓ Comparison report saved to {self.save_dir / 'model_comparison_report.json'}")
        return report
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis between models."""
        improved_dice = self.comparison_results['Improved Model']['dice_scores']
        baseline_dice = self.comparison_results['Baseline Model']['dice_scores']
        
        min_len = min(len(improved_dice), len(baseline_dice))
        improved_dice = improved_dice[:min_len]
        baseline_dice = baseline_dice[:min_len]
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(improved_dice, baseline_dice)
        wilcoxon_stat, wilcoxon_p = stats.mannwhitneyu(improved_dice, baseline_dice, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(improved_dice) + np.var(baseline_dice)) / 2)
        cohens_d = (np.mean(improved_dice) - np.mean(baseline_dice)) / pooled_std
        
        return {
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'mann_whitney_u': {
                'statistic': float(wilcoxon_stat),
                'p_value': float(wilcoxon_p),
                'significant': wilcoxon_p < 0.05
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'descriptive_stats': {
                'mean_difference': float(np.mean(improved_dice) - np.mean(baseline_dice)),
                'median_difference': float(np.median(improved_dice) - np.median(baseline_dice)),
                'std_difference': float(np.std(improved_dice) - np.std(baseline_dice))
            }
        }
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_conclusions(self):
        """Generate conclusions based on comparison results."""
        improved_summary = self.comparison_results['Improved Model']['summary']
        baseline_summary = self.comparison_results['Baseline Model']['summary']
        
        dice_improvement = improved_summary['mean_dice'] - baseline_summary['mean_dice']
        dice_improvement_pct = (dice_improvement / baseline_summary['mean_dice']) * 100
        
        loss_improvement = baseline_summary['mean_loss'] - improved_summary['mean_loss']
        loss_improvement_pct = (loss_improvement / baseline_summary['mean_loss']) * 100
        
        conclusions = {
            'performance_summary': {
                'dice_improvement': float(dice_improvement),
                'dice_improvement_percentage': float(dice_improvement_pct),
                'loss_improvement': float(loss_improvement),
                'loss_improvement_percentage': float(loss_improvement_pct)
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # Generate key findings
        if dice_improvement > 0:
            conclusions['key_findings'].append(f"Improved model shows {dice_improvement:.4f} ({dice_improvement_pct:.1f}%) better Dice score")
        else:
            conclusions['key_findings'].append(f"Baseline model performs better by {-dice_improvement:.4f} ({-dice_improvement_pct:.1f}%) in Dice score")
        
        if loss_improvement > 0:
            conclusions['key_findings'].append(f"Improved model shows {loss_improvement:.4f} ({loss_improvement_pct:.1f}%) better loss")
        
        # Generate recommendations
        if dice_improvement < 0.05:  # Less than 5% improvement
            conclusions['recommendations'].append("Consider further architecture improvements or hyperparameter tuning")
            conclusions['recommendations'].append("Evaluate attention mechanism effectiveness")
        
        if improved_summary['std_dice'] > baseline_summary['std_dice']:
            conclusions['recommendations'].append("Investigate model stability - higher variance in improved model")
        
        return conclusions
    
    def _save_readable_comparison_report(self, report):
        """Save human-readable comparison report."""
        with open(self.save_dir / 'model_comparison_summary.txt', 'w') as f:
            f.write("=" * 80 + "\\n")
            f.write("MODEL COMPARISON REPORT\\n")
            f.write("=" * 80 + "\\n\\n")
            
            f.write(f"Generated: {report['comparison_summary']['timestamp']}\\n")
            f.write(f"Dataset Samples: {report['comparison_summary']['dataset_samples']}\\n")
            f.write(f"Device: {report['comparison_summary']['evaluation_device']}\\n\\n")
            
            # Model Performance Summary
            f.write("MODEL PERFORMANCE SUMMARY:\\n")
            f.write("-" * 40 + "\\n")
            
            for model_name, results in report['model_results'].items():
                f.write(f"\\n{model_name}:\\n")
                f.write(f"  Mean Dice Score: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}\\n")
                f.write(f"  Median Dice Score: {results['median_dice']:.4f}\\n")
                f.write(f"  Range: [{results['min_dice']:.4f}, {results['max_dice']:.4f}]\\n")
                f.write(f"  Mean Loss: {results['mean_loss']:.4f}\\n")
                f.write(f"  Total Samples: {results['total_samples']}\\n")
                f.write(f"  Performance Categories:\\n")
                f.write(f"    Excellent (≥0.8): {results['excellent_samples']} samples\\n")
                f.write(f"    Good (0.6-0.8): {results['good_samples']} samples\\n")
                f.write(f"    Fair (0.4-0.6): {results['fair_samples']} samples\\n")
                f.write(f"    Poor (<0.4): {results['poor_samples']} samples\\n")
            
            # Statistical Analysis
            f.write("\\nSTATISTICAL ANALYSIS:\\n")
            f.write("-" * 40 + "\\n")
            stats = report['statistical_analysis']
            f.write(f"T-test p-value: {stats['t_test']['p_value']:.6f}\\n")
            f.write(f"Statistical significance: {stats['t_test']['significant']}\\n")
            f.write(f"Effect size (Cohen's d): {stats['effect_size']['cohens_d']:.4f} ({stats['effect_size']['interpretation']})\\n")
            f.write(f"Mean difference: {stats['descriptive_stats']['mean_difference']:.4f}\\n")
            
            # Conclusions
            f.write("\\nCONCLUSIONS:\\n")
            f.write("-" * 40 + "\\n")
            conclusions = report['conclusions']
            
            f.write("Key Findings:\\n")
            for finding in conclusions['key_findings']:
                f.write(f"  • {finding}\\n")
            
            f.write("\\nRecommendations:\\n")
            for rec in conclusions['recommendations']:
                f.write(f"  • {rec}\\n")
        
        print(f"✓ Human-readable report saved to {self.save_dir / 'model_comparison_summary.txt'}")

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare improved model with baseline')
    parser.add_argument('--improved-model', type=str, 
                       default='checkpoints_improved/best_model_improved.pth',
                       help='Path to improved model checkpoint')
    parser.add_argument('--split', type=str, choices=['training', 'validation'], 
                       default='validation',
                       help='Dataset split to evaluate on')
    parser.add_argument('--save-dir', type=str, default='model_comparison',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Improved Model: {args.improved_model}")
    print(f"Dataset Split: {args.split}")
    print(f"Save Directory: {args.save_dir}")
    print("-" * 80)
    
    # Initialize comparator
    comparator = ModelComparator(args.improved_model, args.save_dir)
    
    # Load models
    comparator.load_models()
    
    # Load dataset
    dataloader = comparator.load_dataset(args.split)
    
    # Evaluate models
    results = comparator.evaluate_models(dataloader)
    
    # Create visualizations
    comparator.create_comparison_visualizations()
    
    # Generate report
    report = comparator.generate_comparison_report()
    
    # Print summary
    print("\\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    improved_dice = report['model_results']['Improved Model']['mean_dice']
    baseline_dice = report['model_results']['Baseline Model']['mean_dice']
    improvement = improved_dice - baseline_dice
    improvement_pct = (improvement / baseline_dice) * 100
    
    print(f"Improved Model Dice: {improved_dice:.4f}")
    print(f"Baseline Model Dice: {baseline_dice:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"Statistical Significance: {report['statistical_analysis']['t_test']['significant']}")
    print(f"Effect Size: {report['statistical_analysis']['effect_size']['interpretation']}")
    
    print(f"\\n✓ Complete comparison results saved to: {args.save_dir}/")
    print(f"✓ Check {args.save_dir}/model_comparison_summary.txt for detailed analysis")

if __name__ == "__main__":
    main()
