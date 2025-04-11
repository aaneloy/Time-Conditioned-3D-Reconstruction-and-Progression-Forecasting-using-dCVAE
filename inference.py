import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import yaml
import imageio
import glob
import re

from models.dcvae import dCVAE
from data.preprocess import generate_noisy_2d_slice
from utils.seed import set_seed


def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("dcvae_epoch_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"[ERROR] No checkpoints found in {checkpoint_dir}")
    extract_epoch = lambda name: int(re.findall(r"epoch_(\d+)", name)[0])
    latest_file = max(checkpoint_files, key=extract_epoch)
    return os.path.join(checkpoint_dir, latest_file)

def save_inference_visuals(t_months, input_2d, recon_3d, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    input_img = input_2d.detach().squeeze().cpu().numpy()
    recon_vol = recon_3d.detach().squeeze().cpu().numpy()
    mid_slice = recon_vol.shape[0] // 2

    # PNG comparison (color + tight)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(input_img, cmap='viridis')
    axs[0].set_title("Input 2D")
    axs[1].imshow(recon_vol[mid_slice], cmap='viridis')
    axs[1].set_title(f"Reconstructed z={mid_slice}")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"t{t_months}_comparison.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # RGB GIF generation
    gif_frames = []
    for z in range(recon_vol.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(recon_vol[z], cmap='viridis')
        ax.axis('off')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(frame)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f"t{t_months}_reconstruction.gif")
    imageio.mimsave(gif_path, gif_frames, duration=0.15)

def main():
    with open("configs/config.yaml", 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(cfg['seed'])

    model = dCVAE(latent_dim=cfg['model']['latent_dim'],
                  input_channels=1,
                  output_shape=cfg['model']['output_3d_shape']).to(device)

    # Warm-up
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 64, 64).to(device)
        dummy_time = torch.tensor([0.0]).to(device)
        model(dummy_input, dummy_time)

    # Load latest checkpoint
    checkpoint_path = get_latest_checkpoint(cfg['paths']['checkpoints'])
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()

    # Load a sample FLAIR volume
    data_root = cfg['paths']['dataset']
    case_dirs = sorted([d for d in os.listdir(data_root) if d.startswith("BraTS2021_")])
    first_case = os.path.join(data_root, case_dirs[0])
    flair_files = glob.glob(os.path.join(first_case, "*_flair.nii"))
    if not flair_files:
        raise FileNotFoundError(f"[ERROR] No FLAIR file found in {first_case}")
    volume = nib.load(flair_files[0]).get_fdata().astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    # Generate 2D input
    noisy_2d = generate_noisy_2d_slice(volume, resize=(64, 64)).to(device)

    prediction_times = cfg['time_control']['prediction_times_months']
    with torch.no_grad():
        for t_months in prediction_times:
            t = torch.tensor([t_months], dtype=torch.float32).to(device)
            recon_volume, _, _, _ = model(noisy_2d.unsqueeze(0), t)
            recon_volume = torch.sigmoid(recon_volume)

            output_dir = os.path.join("results/inference_outputs", f"t_{t_months}")
            save_inference_visuals(t_months, noisy_2d, recon_volume, output_dir)

if __name__ == "__main__":
    main()
