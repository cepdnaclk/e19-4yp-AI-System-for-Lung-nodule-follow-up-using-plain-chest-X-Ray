# The following python class contains functionality to load DRR + Mask Dataset

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DRRSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_paths = []

        # Traverse all subfolders
        for subdir in os.listdir(root_dir):
            full_subdir = os.path.join(root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for fname in os.listdir(full_subdir):
                if fname.endswith('_drr.png') and not fname.endswith('_drr_mask.png'):
                    drr_path = os.path.join(full_subdir, fname)
                    mask_path = os.path.join(full_subdir, fname.replace('_drr.png', '_drr_mask.png'))
                    if os.path.exists(mask_path):
                        self.image_paths.append((drr_path, mask_path))

        self.transform_img = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(image_size),
            T.ToTensor(),  # Output in [0,1]
            T.Lambda(lambda x: x * 2048 - 1024)  # Rescale to [-1024, +1024]
        ])
        self.transform_mask = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        drr_path, mask_path = self.image_paths[idx]
        drr_img = Image.open(drr_path).convert('L')
        mask_img = Image.open(mask_path).convert('L')

        drr = self.transform_img(drr_img)
        mask = self.transform_mask(mask_img)
        mask = (mask > 0.5).float()  # Ensure binary

        return {
            "xray": drr,      # This will be used as "xray" input in pipeline
            "drr": drr,       
            "mask": mask
        }
