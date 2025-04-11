import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import glob

class BraTS2021Dataset(Dataset):
    def __init__(self, root_dir='data/datasets/brats2021/', modality='flair', transform=None):
        self.root_dir = root_dir
        self.modality = modality.lower()
        self.transform = transform
        self.sample_dirs = []
        self.skipped_cases = []

        all_dirs = sorted([
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d.startswith("BraTS2021_")
        ])

        for d in all_dirs:
            flair_file = glob.glob(os.path.join(d, f"*_{self.modality}.nii*"))
            if len(flair_file) > 0:
                self.sample_dirs.append(d)
            else:
                self.skipped_cases.append(d)

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"[ERROR] No valid cases with '{self.modality}' found in {self.root_dir}")

        print(f"[INFO] Found {len(self.sample_dirs)} valid cases with '{self.modality}'")
        if self.skipped_cases:
            print(f"[WARNING] Skipped {len(self.skipped_cases)} cases (missing {self.modality})")
    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]
        flair_path = glob.glob(os.path.join(sample_path, f"*_{self.modality}.nii*"))

        if len(flair_path) == 0:
            raise FileNotFoundError(f"[ERROR] FLAIR file not found in {sample_path}")

        volume = nib.load(flair_path[0]).get_fdata().astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        volume_tensor = torch.from_numpy(volume).unsqueeze(0)
        volume_tensor = F.interpolate(volume_tensor.unsqueeze(0), size=(64, 64, 64),
                                      mode='trilinear', align_corners=False).squeeze(0)

        if self.transform:
            volume_tensor = self.transform(volume_tensor)

        return volume_tensor
