import torch
import numpy as np
import torch.nn.functional as F

def generate_noisy_2d_slice(volume, slice_idx=None, noise_level=0.2, resize=(64, 64)):
    # Auto-handle volume shapes like [1, D, H, W] or [D, H, W]
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze(0)  # remove batch/channel dim if present
        volume = volume.detach().cpu().numpy()

    # Confirm shape
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (D, H, W), but got shape: {volume.shape}")

    D, H, W = volume.shape

    if slice_idx is None:
        slice_idx = D // 2

    slice_2d = volume[slice_idx, :, :]

    mask = np.random.rand(H, W) > noise_level
    incomplete_slice = slice_2d * mask

    noise = np.random.normal(0, noise_level, (H, W))
    noisy = incomplete_slice + noise
    noisy = np.clip(noisy, 0, 1)

    # [1, 1, H, W] → resized to [1, 64, 64]
    tensor = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=resize, mode='bilinear', align_corners=False)
    return tensor.squeeze(0)  # shape: [1, 64, 64]
