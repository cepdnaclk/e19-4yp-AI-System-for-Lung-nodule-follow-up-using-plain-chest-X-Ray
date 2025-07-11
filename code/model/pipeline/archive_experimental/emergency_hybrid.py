"""
EMERGENCY: Classical + Minimal Deep Learning Hybrid Approach
When deep learning fails, go back to basics with classical methods + minimal ML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import torchxrayvision as xrv
from sklearn.metrics import precision_score, recall_score, f1_score

from drr_dataset_loading import DRRSegmentationDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassicalPreprocessor:
    """Classical image processing for initial filtering."""
    
    def __init__(self):
        pass
    
    def preprocess_drr(self, drr_img):
        """Extract meaningful regions from DRR using classical methods."""
        # Convert to numpy if tensor
        if isinstance(drr_img, torch.Tensor):
            drr_np = drr_img.squeeze().cpu().numpy()
        else:
            drr_np = drr_img
        
        # Normalize to 0-255
        drr_np = ((drr_np - drr_np.min()) / (drr_np.max() - drr_np.min()) * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(drr_np, (5, 5), 0)
        
        # Adaptive thresholding to find potential nodule regions
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours and filter by area and circularity
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for potential nodule regions
        potential_mask = np.zeros_like(drr_np)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 2000:  # Reasonable nodule size range
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Reasonably circular
                        cv2.fillPoly(potential_mask, [contour], 255)
        
        # Convert back to tensor and normalize
        potential_mask = torch.from_numpy(potential_mask / 255.0).float()
        return potential_mask

class MinimalClassifier(nn.Module):
    """Extremely simple classifier - just a few layers."""
    
    def __init__(self):
        super().__init__()
        
        # Simple feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        classification = self.classifier(features)
        return classification

class HybridModel(nn.Module):
    """Hybrid classical + minimal ML model."""
    
    def __init__(self):
        super().__init__()
        self.preprocessor = ClassicalPreprocessor()
        self.classifier = MinimalClassifier()
        
    def forward(self, xray, drr):
        batch_size = drr.shape[0]
        height, width = drr.shape[2], drr.shape[3]
        
        # Process each sample in the batch
        final_masks = []
        
        for i in range(batch_size):
            # Classical preprocessing of DRR
            classical_mask = self.preprocessor.preprocess_drr(drr[i])
            
            if classical_mask.sum() == 0:
                # No potential regions found - return empty mask
                final_masks.append(torch.zeros(1, height, width))
            else:
                # Resize for classifier input
                classical_resized = F.interpolate(
                    classical_mask.unsqueeze(0).unsqueeze(0), 
                    size=(256, 256), 
                    mode='nearest'
                )
                
                # Simple classification
                confidence = self.classifier(classical_resized)
                
                # Apply confidence to classical mask
                final_mask = classical_mask * confidence.item()
                final_masks.append(final_mask.unsqueeze(0))
        
        # Stack batch
        final_output = torch.stack(final_masks, dim=0)
        
        return final_output, final_output  # Return same tensor twice for compatibility

def train_hybrid_model():
    """Train the hybrid classical + ML model."""
    
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (512, 512),
        'batch_size': 4,
        'learning_rate': 1e-3,
        'num_epochs': 20,
        'save_dir': './checkpoints_hybrid',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    logger.info("Starting hybrid classical + ML training")
    
    # Load dataset
    try:
        dataset = DRRSegmentationDataset(
            root_dir=config['data_root'],
            image_size=config['image_size'],
            augment=False
        )
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Use smaller subset for faster development
        if len(dataset) > 100:
            subset_indices = torch.randperm(len(dataset))[:100]
            dataset = torch.utils.data.Subset(dataset, subset_indices)
            logger.info(f"Using subset of {len(dataset)} samples")
        
        # Train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=0)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    # Initialize hybrid model
    model = HybridModel()
    model = model.to(config['device'])
    
    # Only train the classifier part
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_precision = 0.0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                xray = batch["xray"].to(config['device'])
                drr = batch["drr"].to(config['device'])
                mask = batch["mask"].to(config['device'])

                optimizer.zero_grad()

                # Forward pass
                seg_out, _ = model(xray, drr)
                
                # Simple BCE loss
                loss = F.binary_cross_entropy(seg_out, mask)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 5 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                logger.error(f"Error in training: {e}")
                continue
        
        # Validation phase
        model.eval()
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    xray = batch["xray"].to(config['device'])
                    drr = batch["drr"].to(config['device'])
                    mask = batch["mask"].to(config['device'])

                    seg_out, _ = model(xray, drr)
                    
                    all_preds.append(seg_out.cpu())
                    all_masks.append(mask.cpu())
                    
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
        
        # Calculate metrics
        if all_preds and all_masks:
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            
            # Test different thresholds
            best_precision_score = 0.0
            best_threshold = 0.5
            
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                pred_binary = (all_preds > threshold).float()
                
                if pred_binary.sum() > 0:  # Avoid division by zero
                    pred_flat = pred_binary.view(-1).numpy()
                    mask_flat = all_masks.view(-1).numpy()
                    
                    precision = precision_score(mask_flat, pred_flat, zero_division=0)
                    recall = recall_score(mask_flat, pred_flat, zero_division=0)
                    
                    if precision > best_precision_score:
                        best_precision_score = precision
                        best_threshold = threshold
            
            logger.info(f"Epoch {epoch+1}: Best Precision: {best_precision_score:.4f} "
                       f"at threshold {best_threshold:.1f}")
            
            # Save if improved
            if best_precision_score > best_precision:
                best_precision = best_precision_score
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_precision': best_precision,
                    'best_threshold': best_threshold,
                    'config': config
                }
                
                torch.save(checkpoint, os.path.join(config['save_dir'], 'best_hybrid_model.pth'))
                logger.info(f"New best precision saved: {best_precision:.4f}")
        
        scheduler.step()
    
    logger.info(f"Training completed. Best precision: {best_precision:.4f}")
    return best_precision, best_threshold

def test_classical_only():
    """Test purely classical approach without any ML."""
    
    logger.info("Testing purely classical approach...")
    
    config = {
        'data_root': '../../../DRR dataset/LIDC_LDRI',
        'image_size': (512, 512),
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    # Load small dataset for testing
    dataset = DRRSegmentationDataset(
        root_dir=config['data_root'],
        image_size=config['image_size'],
        augment=False
    )
    
    # Test on first 20 samples
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    preprocessor = ClassicalPreprocessor()
    all_precisions = []
    
    for i, batch in enumerate(test_loader):
        if i >= 20:
            break
            
        drr = batch["drr"]
        mask = batch["mask"]
        
        # Apply classical preprocessing
        classical_mask = preprocessor.preprocess_drr(drr[0])
        
        # Calculate precision
        if classical_mask.sum() > 0:
            pred_flat = classical_mask.view(-1).numpy()
            mask_flat = mask.view(-1).numpy()
            
            precision = precision_score(mask_flat, pred_flat, zero_division=0)
            all_precisions.append(precision)
            
            logger.info(f"Sample {i+1}: Precision = {precision:.4f}")
    
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    logger.info(f"Classical-only average precision: {avg_precision:.4f}")
    
    return avg_precision

if __name__ == "__main__":
    print("="*60)
    print("EMERGENCY HYBRID APPROACH: Classical + Minimal ML")
    print("="*60)
    
    # Test classical approach first
    print("\n1. Testing purely classical approach...")
    classical_precision = test_classical_only()
    
    # Train hybrid model
    print("\n2. Training hybrid classical + ML model...")
    hybrid_precision, threshold = train_hybrid_model()
    
    print(f"\nFINAL RESULTS:")
    print(f"Classical-only precision: {classical_precision:.4f}")
    print(f"Hybrid precision: {hybrid_precision:.4f}")
    print(f"Best threshold: {threshold:.1f}")
    
    if hybrid_precision > 0.05:
        print("✅ SUCCESS: Achieved meaningful precision!")
    else:
        print("❌ Still struggling - may need data quality investigation")
