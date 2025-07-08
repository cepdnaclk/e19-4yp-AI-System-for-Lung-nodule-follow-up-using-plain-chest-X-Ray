# Enhanced DRR Dataset Loading with improved error handling and data augmentation

import os
import logging
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import random

logger = logging.getLogger(__name__)

class DRRSegmentationDataset(Dataset):
    """Enhanced DRR Segmentation Dataset with data augmentation and validation."""
    
    def __init__(self, root_dir, image_size=(512, 512), augment=False, normalize=True):
        """
        Args:
            root_dir (str): Root directory containing subfolders of DRR images and masks.
            image_size (tuple): Desired output size (height, width) for images and masks.
            augment (bool): Whether to apply data augmentation.
            normalize (bool): Whether to normalize images to torchxrayvision format.
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.image_paths = []
        
        self._load_image_paths()
        self._setup_transforms()
        
        logger.info(f"Loaded {len(self.image_paths)} image-mask pairs from {root_dir}")

    def _load_image_paths(self):
        """Load all valid image-mask pairs from the dataset directory."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        # Traverse all subfolders in the root directory
        for subdir in os.listdir(self.root_dir):
            full_subdir = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
                
            # Look for DRR images and corresponding mask files
            for fname in os.listdir(full_subdir):
                if fname.endswith('_drr.png') and not fname.endswith('_drr_mask.png'):
                    drr_path = os.path.join(full_subdir, fname)
                    mask_path = os.path.join(full_subdir, fname.replace('_drr.png', '_drr_mask.png'))
                    
                    # Validate both files exist and are readable
                    if os.path.exists(mask_path):
                        try:
                            # Quick validation that files can be opened
                            with Image.open(drr_path) as img:
                                img.verify()
                            with Image.open(mask_path) as img:
                                img.verify()
                            self.image_paths.append((drr_path, mask_path))
                        except Exception as e:
                            logger.warning(f"Skipping corrupted file pair {drr_path}: {e}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image-mask pairs found in {self.root_dir}")

    def _setup_transforms(self):
        """Setup image transformations."""
        # Base transforms
        base_transforms = [
            T.Grayscale(num_output_channels=1),
            T.Resize(self.image_size),
        ]
        
        # Augmentation transforms (applied randomly)
        if self.augment:
            self.augment_transforms = T.Compose([
                T.RandomRotation(degrees=10),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        
        # Final transforms
        final_transforms = [T.ToTensor()]
        
        if self.normalize:
            # Rescale to [-1024, 1024] for torchxrayvision compatibility
            final_transforms.append(T.Lambda(lambda x: x * 2048 - 1024))
        
        self.transform_img = T.Compose(base_transforms + final_transforms)
        self.transform_mask = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor()
        ])

    def _apply_augmentation(self, drr_img, mask_img):
        """Apply synchronized augmentation to both image and mask."""
        if not self.augment:
            return drr_img, mask_img
        
        # Set random seed for synchronized transforms
        seed = random.randint(0, 2**32)
        
        # Apply same random transforms to both image and mask
        random.seed(seed)
        torch.manual_seed(seed)
        drr_aug = self.augment_transforms(drr_img)
        
        random.seed(seed)
        torch.manual_seed(seed)
        mask_aug = self.augment_transforms(mask_img)
        
        return drr_aug, mask_aug

    def __len__(self):
        """Return the number of image-mask pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        try:
            # Get the DRR and mask file paths for the given index
            drr_path, mask_path = self.image_paths[idx]
            
            # Open images in grayscale mode
            drr_img = Image.open(drr_path).convert('L')
            mask_img = Image.open(mask_path).convert('L')
            
            # Apply augmentation if enabled
            drr_img, mask_img = self._apply_augmentation(drr_img, mask_img)
            
            # Apply transformations
            drr = self.transform_img(drr_img)
            mask = self.transform_mask(mask_img)
            
            # Ensure binary mask
            mask = (mask > 0.5).float()
            
            # Validate tensor shapes
            assert drr.shape[1:] == self.image_size, f"DRR shape mismatch: {drr.shape}"
            assert mask.shape[1:] == self.image_size, f"Mask shape mismatch: {mask.shape}"
            
            # Return a dictionary with the processed images and mask
            return {
                "xray": drr,      # Using DRR as xray input for consistency
                "drr": drr,       # Also keeping as DRR for attention computation
                "mask": mask,
                "filename": os.path.basename(drr_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({self.image_paths[idx]}): {e}")
            # Return a dummy sample to avoid breaking the training loop
            dummy_tensor = torch.zeros((1, *self.image_size))
            return {
                "xray": dummy_tensor,
                "drr": dummy_tensor,
                "mask": dummy_tensor,
                "filename": "error_sample"
            }

    def get_sample_by_filename(self, filename):
        """Get a specific sample by filename."""
        for idx, (drr_path, _) in enumerate(self.image_paths):
            if os.path.basename(drr_path) == filename:
                return self.__getitem__(idx)
        raise ValueError(f"Sample with filename {filename} not found")

    def get_statistics(self):
        """Get dataset statistics."""
        stats = {
            'num_samples': len(self.image_paths),
            'image_size': self.image_size,
            'augmentation': self.augment,
            'normalization': self.normalize
        }
        
        # Sample a few images to get pixel statistics
        if len(self.image_paths) > 0:
            sample_indices = np.random.choice(
                min(len(self.image_paths), 100), 
                size=min(10, len(self.image_paths)), 
                replace=False
            )
            
            pixel_values = []
            mask_ratios = []
            
            for idx in sample_indices:
                sample = self.__getitem__(idx)
                pixel_values.extend(sample['xray'].flatten().tolist())
                mask_ratios.append(sample['mask'].mean().item())
            
            stats.update({
                'pixel_mean': np.mean(pixel_values),
                'pixel_std': np.std(pixel_values),
                'pixel_min': np.min(pixel_values),
                'pixel_max': np.max(pixel_values),
                'avg_mask_ratio': np.mean(mask_ratios),
                'mask_ratio_std': np.std(mask_ratios)
            })
        
        return stats
