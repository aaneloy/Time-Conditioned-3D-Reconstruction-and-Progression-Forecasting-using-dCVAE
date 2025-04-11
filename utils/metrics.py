import torch
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim
import numpy as np


def mse_metric(output, target):
    """
    Mean Squared Error between output and target.
    Assumes output and target are torch tensors.
    """
    return torch.mean((output - target) ** 2).item()


def ssim_metric(output, target):
    """
    Structural Similarity Index (SSIM) over 2D slices.
    Works only on [B, 1, D, H, W] inputs with B=1.
    """
    output_np = output.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()

    if output_np.ndim == 3:
        # Compute SSIM over middle slice
        d = output_np.shape[0] // 2
        return ssim(output_np[d], target_np[d], data_range=1.0)
    else:
        raise ValueError("Expected input shape [1, 1, D, H, W] or [1, D, H, W]")


def auc_metric(output, target):
    """
    Compute ROC-AUC for binarized prediction vs. ground truth.
    Flattens the tensors and treats values > 0.5 as class 1.
    """
    output_np = output.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()

    try:
        return roc_auc_score((target_np > 0.5).astype(int), output_np)
    except ValueError:
        # If there's only one class in the ground truth, AUC is undefined
        return float('nan')
