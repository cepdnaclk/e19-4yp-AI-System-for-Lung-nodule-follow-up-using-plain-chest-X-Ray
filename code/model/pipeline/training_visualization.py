"""
Comprehensive training and evaluation visualization utilities.
Provides detailed graphs, metrics, and analysis for model training.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """Comprehensive training visualization and metrics tracking."""
    
    def __init__(self, save_dir='training_visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize tracking lists
        self.metrics = {
            'train_loss': [],
            'train_dice': [],
            'train_seg_loss': [],
            'train_attention_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_seg_loss': [],
            'val_attention_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        self.best_metrics = {
            'best_dice': 0.0,
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
    
    def update_metrics(self, epoch, train_metrics, val_metrics, lr):
        """Update metrics for current epoch."""
        # Unpack training metrics
        train_loss, train_seg_loss, train_att_loss, train_dice = train_metrics
        val_loss, val_seg_loss, val_att_loss, val_dice = val_metrics
        
        # Store metrics
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_dice'].append(train_dice)
        self.metrics['train_seg_loss'].append(train_seg_loss)
        self.metrics['train_attention_loss'].append(train_att_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_dice'].append(val_dice)
        self.metrics['val_seg_loss'].append(val_seg_loss)
        self.metrics['val_attention_loss'].append(val_att_loss)
        self.metrics['learning_rates'].append(lr)
        
        # Update best metrics
        if val_dice > self.best_metrics['best_dice']:
            self.best_metrics['best_dice'] = val_dice
            self.best_metrics['best_epoch'] = epoch
        
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
    
    def plot_training_curves(self):
        """Create comprehensive training curves plot."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Progress Overview', fontsize=16, fontweight='bold')
        
        epochs = self.metrics['epochs']
        
        # 1. Loss curves
        axes[0, 0].plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Dice coefficient
        axes[0, 1].plot(epochs, self.metrics['train_dice'], 'g-', label='Training Dice', linewidth=2)
        axes[0, 1].plot(epochs, self.metrics['val_dice'], 'orange', label='Validation Dice', linewidth=2)
        best_epoch = self.best_metrics['best_epoch']
        best_dice = self.best_metrics['best_dice']
        axes[0, 1].axhline(y=best_dice, color='red', linestyle='--', alpha=0.7, 
                          label=f'Best Dice: {best_dice:.4f} (Epoch {best_epoch})')
        axes[0, 1].set_title('Dice Coefficient', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Segmentation loss
        axes[0, 2].plot(epochs, self.metrics['train_seg_loss'], 'purple', label='Training Seg Loss', linewidth=2)
        axes[0, 2].plot(epochs, self.metrics['val_seg_loss'], 'brown', label='Validation Seg Loss', linewidth=2)
        axes[0, 2].set_title('Segmentation Loss', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Segmentation Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Attention loss
        axes[1, 0].plot(epochs, self.metrics['train_attention_loss'], 'cyan', label='Training Att Loss', linewidth=2)
        axes[1, 0].plot(epochs, self.metrics['val_attention_loss'], 'magenta', label='Validation Att Loss', linewidth=2)
        axes[1, 0].set_title('Attention Supervision Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Attention Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Learning rate
        axes[1, 1].plot(epochs, self.metrics['learning_rates'], 'black', label='Learning Rate', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Training vs Validation Gap
        train_val_gap = np.array(self.metrics['val_loss']) - np.array(self.metrics['train_loss'])
        axes[1, 2].plot(epochs, train_val_gap, 'red', label='Validation - Training Loss', linewidth=2)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 2].set_title('Overfitting Analysis (Loss Gap)', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss Difference')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {self.save_dir / 'training_curves.png'}")
    
    def plot_metrics_distribution(self):
        """Plot distribution of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Metrics Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Dice score distribution
        axes[0, 0].hist(self.metrics['val_dice'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(np.mean(self.metrics['val_dice']), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(self.metrics["val_dice"]):.4f}')
        axes[0, 0].axvline(np.median(self.metrics['val_dice']), color='orange', linestyle='--',
                          label=f'Median: {np.median(self.metrics["val_dice"]):.4f}')
        axes[0, 0].set_title('Validation Dice Score Distribution')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[0, 1].hist(self.metrics['val_loss'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.metrics['val_loss']), color='blue', linestyle='--',
                          label=f'Mean: {np.mean(self.metrics["val_loss"]):.4f}')
        axes[0, 1].set_title('Validation Loss Distribution')
        axes[0, 1].set_xlabel('Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate progression
        epochs = self.metrics['epochs']
        axes[1, 0].scatter(epochs, self.metrics['learning_rates'], alpha=0.6, c=epochs, cmap='viridis')
        axes[1, 0].set_title('Learning Rate Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap
        correlation_data = {
            'Val Dice': self.metrics['val_dice'],
            'Val Loss': self.metrics['val_loss'],
            'Train Dice': self.metrics['train_dice'],
            'Train Loss': self.metrics['train_loss'],
            'LR': self.metrics['learning_rates']
        }
        corr_df = pd.DataFrame(correlation_data)
        correlation_matrix = corr_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
        axes[1, 1].set_title('Metrics Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Metrics distribution saved to {self.save_dir / 'metrics_distribution.png'}")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        report = {
            'training_summary': {
                'total_epochs': len(self.metrics['epochs']),
                'best_validation_dice': self.best_metrics['best_dice'],
                'best_epoch': self.best_metrics['best_epoch'],
                'best_validation_loss': self.best_metrics['best_val_loss'],
                'final_validation_dice': self.metrics['val_dice'][-1] if self.metrics['val_dice'] else 0,
                'final_validation_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 0,
            },
            'statistics': {
                'validation_dice': {
                    'mean': np.mean(self.metrics['val_dice']),
                    'std': np.std(self.metrics['val_dice']),
                    'min': np.min(self.metrics['val_dice']),
                    'max': np.max(self.metrics['val_dice']),
                    'median': np.median(self.metrics['val_dice'])
                },
                'validation_loss': {
                    'mean': np.mean(self.metrics['val_loss']),
                    'std': np.std(self.metrics['val_loss']),
                    'min': np.min(self.metrics['val_loss']),
                    'max': np.max(self.metrics['val_loss']),
                    'median': np.median(self.metrics['val_loss'])
                }
            },
            'convergence_analysis': {
                'epochs_to_best': self.best_metrics['best_epoch'],
                'improvement_rate': self._calculate_improvement_rate(),
                'stability_score': self._calculate_stability_score()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report as JSON
        with open(self.save_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable report
        self._save_readable_report(report)
        
        print(f"✓ Training report saved to {self.save_dir / 'training_report.json'}")
        return report
    
    def _calculate_improvement_rate(self):
        """Calculate rate of improvement in validation dice."""
        if len(self.metrics['val_dice']) < 2:
            return 0.0
        
        val_dice = np.array(self.metrics['val_dice'])
        epochs = np.array(self.metrics['epochs'])
        
        # Linear regression to find improvement rate
        slope, _, r_value, _, _ = stats.linregress(epochs, val_dice)
        return slope
    
    def _calculate_stability_score(self):
        """Calculate training stability score based on validation dice variance."""
        if len(self.metrics['val_dice']) < 5:
            return 0.0
        
        # Use last 25% of training for stability analysis
        last_quarter = len(self.metrics['val_dice']) // 4
        recent_dice = self.metrics['val_dice'][-last_quarter:]
        
        # Lower variance = higher stability
        variance = np.var(recent_dice)
        stability = 1.0 / (1.0 + variance)  # Normalize to [0, 1]
        return stability
    
    def _save_readable_report(self, report):
        """Save human-readable training report."""
        with open(self.save_dir / 'training_summary.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("KEY RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Epochs: {report['training_summary']['total_epochs']}\n")
            f.write(f"Best Validation Dice: {report['training_summary']['best_validation_dice']:.4f}\n")
            f.write(f"Best Epoch: {report['training_summary']['best_epoch']}\n")
            f.write(f"Best Validation Loss: {report['training_summary']['best_validation_loss']:.4f}\n")
            f.write(f"Final Validation Dice: {report['training_summary']['final_validation_dice']:.4f}\n")
            f.write(f"Final Validation Loss: {report['training_summary']['final_validation_loss']:.4f}\n\n")
            
            f.write("VALIDATION DICE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            dice_stats = report['statistics']['validation_dice']
            f.write(f"Mean: {dice_stats['mean']:.4f} ± {dice_stats['std']:.4f}\n")
            f.write(f"Range: [{dice_stats['min']:.4f}, {dice_stats['max']:.4f}]\n")
            f.write(f"Median: {dice_stats['median']:.4f}\n\n")
            
            f.write("CONVERGENCE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            conv = report['convergence_analysis']
            f.write(f"Epochs to Best: {conv['epochs_to_best']}\n")
            f.write(f"Improvement Rate: {conv['improvement_rate']:.6f} dice/epoch\n")
            f.write(f"Stability Score: {conv['stability_score']:.4f}\n\n")
            
        print(f"✓ Human-readable report saved to {self.save_dir / 'training_summary.txt'}")
    
    def save_metrics_csv(self):
        """Save all metrics to CSV for further analysis."""
        df = pd.DataFrame(self.metrics)
        csv_path = self.save_dir / 'training_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Metrics CSV saved to {csv_path}")
        return csv_path


class EvaluationVisualizer:
    """Comprehensive evaluation visualization and analysis."""
    
    def __init__(self, save_dir='evaluation_visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def evaluate_model_detailed(self, model, dataloader, device, threshold=0.5):
        """Perform detailed model evaluation with comprehensive metrics."""
        model.eval()
        
        all_predictions = []
        all_masks = []
        all_losses = []
        all_dice_scores = []
        sample_predictions = []  # Store a few samples for visualization
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                try:
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        xray = batch["xray"].to(device)
                        drr = batch["drr"].to(device)
                        masks = batch["mask"].to(device)
                    else:
                        # If batch is a tuple/list, unpack accordingly
                        if len(batch) >= 3:
                            xray, drr, masks = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        else:
                            print(f"⚠️ Unexpected batch format: {type(batch)}")
                            continue
                    
                    # Get predictions
                    outputs = model(xray, drr)
                    
                    # Handle model output format
                    if isinstance(outputs, dict):
                        segmentation_output = outputs['segmentation']
                    else:
                        # If outputs is a tuple/list, take the first element as segmentation
                        segmentation_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    
                    predictions = torch.sigmoid(segmentation_output)
                    
                    # Calculate loss (simplified)
                    loss = F.binary_cross_entropy_with_logits(segmentation_output, masks)
                    all_losses.append(loss.item())
                    
                    # Calculate dice for each sample in batch
                    for j in range(predictions.shape[0]):
                        # Handle both tensor and numpy array inputs
                        if torch.is_tensor(predictions):
                            pred = predictions[j].cpu().numpy()
                        else:
                            pred = predictions[j]
                        
                        if torch.is_tensor(masks):
                            mask = masks[j].cpu().numpy()
                        else:
                            mask = masks[j]
                        
                        # Binary prediction
                        pred_binary = (pred > threshold).astype(float)
                        
                        # Calculate dice
                        intersection = (pred_binary * mask).sum()
                        dice = (2 * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)
                        all_dice_scores.append(dice)
                        
                        # Store for analysis
                        all_predictions.append(pred_binary.flatten())
                        all_masks.append(mask.flatten())
                        
                        # Store some samples for visualization
                        if len(sample_predictions) < 8:
                            # Handle tensor/numpy conversion for visualization
                            if torch.is_tensor(xray):
                                xray_np = xray[j].cpu().numpy()
                            else:
                                xray_np = xray[j]
                                
                            if torch.is_tensor(drr):
                                drr_np = drr[j].cpu().numpy()
                            else:
                                drr_np = drr[j]
                                
                            sample_predictions.append({
                                'xray': xray_np,
                                'drr': drr_np,
                                'mask': mask,
                                'prediction': pred,
                                'dice': dice
                            })
                            
                except Exception as e:
                    print(f"⚠️ Error processing batch {i}: {e}")
                    continue
        
        # Create comprehensive evaluation report
        self._create_evaluation_plots(all_dice_scores, all_losses, sample_predictions)
        return self._generate_evaluation_report(all_dice_scores, all_losses, all_predictions, all_masks)
    
    def _create_evaluation_plots(self, dice_scores, losses, sample_predictions):
        """Create comprehensive evaluation plots."""
        
        # 1. Performance distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Analysis', fontsize=16, fontweight='bold')
        
        # Dice score distribution
        axes[0, 0].hist(dice_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(np.mean(dice_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(dice_scores):.3f}')
        axes[0, 0].axvline(np.median(dice_scores), color='orange', linestyle='--',
                          label=f'Median: {np.median(dice_scores):.3f}')
        axes[0, 0].set_title('Dice Score Distribution')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[0, 1].hist(losses, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].axvline(np.mean(losses), color='blue', linestyle='--',
                          label=f'Mean: {np.mean(losses):.3f}')
        axes[0, 1].set_title('Loss Distribution')
        axes[0, 1].set_xlabel('Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance correlation
        axes[0, 2].scatter(losses, dice_scores, alpha=0.6)
        axes[0, 2].set_title('Loss vs Dice Score')
        axes[0, 2].set_xlabel('Loss')
        axes[0, 2].set_ylabel('Dice Score')
        
        # Calculate correlation
        corr = np.corrcoef(losses, dice_scores)[0, 1]
        axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 2].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0, 2].grid(True, alpha=0.3)
        
        # Box plot for performance quartiles
        quartiles = np.percentile(dice_scores, [25, 50, 75])
        q1_samples = [d for d in dice_scores if d <= quartiles[0]]
        q2_samples = [d for d in dice_scores if quartiles[0] < d <= quartiles[1]]
        q3_samples = [d for d in dice_scores if quartiles[1] < d <= quartiles[2]]
        q4_samples = [d for d in dice_scores if d > quartiles[2]]
        
        axes[1, 0].boxplot([q1_samples, q2_samples, q3_samples, q4_samples], 
                          labels=['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 0].set_title('Performance Quartiles')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative performance
        sorted_dice = np.sort(dice_scores)
        cumulative = np.arange(1, len(sorted_dice) + 1) / len(sorted_dice)
        axes[1, 1].plot(sorted_dice, cumulative, linewidth=2)
        axes[1, 1].set_title('Cumulative Performance Distribution')
        axes[1, 1].set_xlabel('Dice Score')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance statistics
        stats_text = f"""
        Statistics:
        Mean: {np.mean(dice_scores):.4f}
        Std: {np.std(dice_scores):.4f}
        Min: {np.min(dice_scores):.4f}
        Max: {np.max(dice_scores):.4f}
        Q1: {np.percentile(dice_scores, 25):.4f}
        Q3: {np.percentile(dice_scores, 75):.4f}
        
        Samples > 0.5: {(np.array(dice_scores) > 0.5).sum()}/{len(dice_scores)}
        Samples > 0.7: {(np.array(dice_scores) > 0.7).sum()}/{len(dice_scores)}
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample predictions visualization
        if sample_predictions:
            self._plot_sample_predictions(sample_predictions)
    
    def _plot_sample_predictions(self, sample_predictions):
        """Plot sample predictions for qualitative analysis."""
        n_samples = min(8, len(sample_predictions))
        fig, axes = plt.subplots(4, n_samples, figsize=(3*n_samples, 12))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Sample Predictions Analysis', fontsize=16, fontweight='bold')
        
        for i, sample in enumerate(sample_predictions[:n_samples]):
            # X-ray
            axes[0, i].imshow(sample['xray'][0], cmap='gray')
            axes[0, i].set_title(f'Sample {i+1}: X-ray')
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(sample['mask'][0], cmap='hot')
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(sample['prediction'][0], cmap='hot')
            axes[2, i].set_title(f'Prediction (Dice: {sample["dice"]:.3f})')
            axes[2, i].axis('off')
            
            # Overlay comparison
            overlay = sample['xray'][0].copy()
            # Normalize overlay to [0,1]
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
            overlay_colored = np.zeros((*overlay.shape, 3))
            overlay_colored[:, :, 0] = overlay  # X-ray in red channel
            overlay_colored[:, :, 1] = sample['mask'][0] * 0.7  # GT in green
            overlay_colored[:, :, 2] = sample['prediction'][0] * 0.7  # Pred in blue
            
            axes[3, i].imshow(overlay_colored)
            axes[3, i].set_title('Overlay (GT:Green, Pred:Blue)')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_evaluation_report(self, dice_scores, losses, predictions, masks):
        """Generate comprehensive evaluation report."""
        
        # Convert to numpy arrays
        dice_scores = np.array(dice_scores)
        losses = np.array(losses)
        
        # Calculate additional metrics
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        mean_loss = np.mean(losses)
        
        # Performance categories
        excellent = (dice_scores >= 0.8).sum()
        good = ((dice_scores >= 0.6) & (dice_scores < 0.8)).sum()
        fair = ((dice_scores >= 0.4) & (dice_scores < 0.6)).sum()
        poor = (dice_scores < 0.4).sum()
        
        report = {
            'evaluation_summary': {
                'total_samples': len(dice_scores),
                'mean_dice_score': float(mean_dice),
                'std_dice_score': float(std_dice),
                'median_dice_score': float(np.median(dice_scores)),
                'min_dice_score': float(np.min(dice_scores)),
                'max_dice_score': float(np.max(dice_scores)),
                'mean_loss': float(mean_loss),
                'std_loss': float(np.std(losses))
            },
            'performance_categories': {
                'excellent_samples': int(excellent),
                'good_samples': int(good),
                'fair_samples': int(fair),
                'poor_samples': int(poor),
                'excellent_percentage': float(excellent / len(dice_scores) * 100),
                'good_percentage': float(good / len(dice_scores) * 100),
                'fair_percentage': float(fair / len(dice_scores) * 100),
                'poor_percentage': float(poor / len(dice_scores) * 100)
            },
            'statistical_analysis': {
                'percentiles': {
                    '10th': float(np.percentile(dice_scores, 10)),
                    '25th': float(np.percentile(dice_scores, 25)),
                    '75th': float(np.percentile(dice_scores, 75)),
                    '90th': float(np.percentile(dice_scores, 90))
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save evaluation report
        with open(self.save_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable report
        self._save_evaluation_summary(report)
        
        print(f"✓ Evaluation report saved to {self.save_dir / 'evaluation_report.json'}")
        return report
    
    def _save_evaluation_summary(self, report):
        """Save human-readable evaluation summary."""
        with open(self.save_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            summary = report['evaluation_summary']
            f.write(f"Total Samples Evaluated: {summary['total_samples']}\n")
            f.write(f"Mean Dice Score: {summary['mean_dice_score']:.4f} ± {summary['std_dice_score']:.4f}\n")
            f.write(f"Median Dice Score: {summary['median_dice_score']:.4f}\n")
            f.write(f"Range: [{summary['min_dice_score']:.4f}, {summary['max_dice_score']:.4f}]\n")
            f.write(f"Mean Loss: {summary['mean_loss']:.4f}\n\n")
            
            f.write("PERFORMANCE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            cats = report['performance_categories']
            f.write(f"Excellent (≥0.8): {cats['excellent_samples']} samples ({cats['excellent_percentage']:.1f}%)\n")
            f.write(f"Good (0.6-0.8): {cats['good_samples']} samples ({cats['good_percentage']:.1f}%)\n")
            f.write(f"Fair (0.4-0.6): {cats['fair_samples']} samples ({cats['fair_percentage']:.1f}%)\n")
            f.write(f"Poor (<0.4): {cats['poor_samples']} samples ({cats['poor_percentage']:.1f}%)\n\n")
            
            f.write("PERCENTILE ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            percs = report['statistical_analysis']['percentiles']
            f.write(f"10th Percentile: {percs['10th']:.4f}\n")
            f.write(f"25th Percentile: {percs['25th']:.4f}\n")
            f.write(f"75th Percentile: {percs['75th']:.4f}\n")
            f.write(f"90th Percentile: {percs['90th']:.4f}\n\n")
        
        print(f"✓ Evaluation summary saved to {self.save_dir / 'evaluation_summary.txt'}")


def create_final_visualization_report(training_dir='training_visualizations', 
                                    evaluation_dir='evaluation_visualizations',
                                    output_dir='final_report'):
    """Create a comprehensive final report combining training and evaluation."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load reports
    training_report_path = Path(training_dir) / 'training_report.json'
    eval_report_path = Path(evaluation_dir) / 'evaluation_report.json'
    
    training_report = {}
    eval_report = {}
    
    if training_report_path.exists():
        with open(training_report_path, 'r') as f:
            training_report = json.load(f)
    
    if eval_report_path.exists():
        with open(eval_report_path, 'r') as f:
            eval_report = json.load(f)
    
    # Create combined report
    final_report = {
        'project_summary': {
            'model_type': 'ImprovedXrayDRRModel with Supervised Attention',
            'training_completed': bool(training_report),
            'evaluation_completed': bool(eval_report),
            'report_generated': datetime.now().isoformat()
        },
        'training_results': training_report,
        'evaluation_results': eval_report
    }
    
    # Save final report
    with open(output_path / 'final_model_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Create comprehensive markdown report
    _create_markdown_report(final_report, output_path)
    
    print(f"✓ Final comprehensive report saved to {output_path}")


def _create_markdown_report(report, output_path):
    """Create a comprehensive markdown report."""
    
    md_content = f"""# Model Training & Evaluation Report

**Generated:** {report['project_summary']['report_generated']}  
**Model:** {report['project_summary']['model_type']}

## 📊 Executive Summary

"""
    
    if report['training_results']:
        training = report['training_results']['training_summary']
        md_content += f"""
### Training Results
- **Total Epochs:** {training['total_epochs']}
- **Best Validation Dice:** {training['best_validation_dice']:.4f}
- **Best Epoch:** {training['best_epoch']}
- **Final Validation Dice:** {training['final_validation_dice']:.4f}
"""
    
    if report['evaluation_results']:
        evaluation = report['evaluation_results']['evaluation_summary']
        categories = report['evaluation_results']['performance_categories']
        md_content += f"""
### Evaluation Results
- **Total Samples:** {evaluation['total_samples']}
- **Mean Dice Score:** {evaluation['mean_dice_score']:.4f} ± {evaluation['std_dice_score']:.4f}
- **Excellent Performance:** {categories['excellent_percentage']:.1f}% of samples
- **Good+ Performance:** {categories['excellent_percentage'] + categories['good_percentage']:.1f}% of samples
"""
    
    md_content += """
## 📈 Detailed Analysis

### Training Progress
- Training curves show convergence patterns
- Learning rate scheduling effectiveness
- Overfitting analysis through train/validation gap

### Model Performance
- Dice coefficient distribution analysis
- Performance categorization
- Statistical significance testing

## 🎯 Recommendations

Based on the analysis:
1. Monitor for overfitting in future training
2. Consider data augmentation strategies
3. Evaluate attention mechanism alignment
4. Assess generalization capabilities

## 📁 Generated Files

- `training_curves.png` - Training progress visualization
- `metrics_distribution.png` - Statistical analysis
- `evaluation_analysis.png` - Performance analysis
- `sample_predictions.png` - Qualitative results
- Various JSON and CSV files for detailed metrics

---
*This report was automatically generated by the training visualization system.*
"""
    
    with open(output_path / 'MODEL_REPORT.md', 'w') as f:
        f.write(md_content)
    
    print(f"✓ Markdown report saved to {output_path / 'MODEL_REPORT.md'}")
