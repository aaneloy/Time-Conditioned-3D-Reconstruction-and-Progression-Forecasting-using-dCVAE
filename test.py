import torch
from torch.utils.data import DataLoader
from models.dcvae import dCVAE
from data.dataloader import BraTS2021Dataset
from data.preprocess import generate_noisy_2d_slice
from utils.metrics import mse_metric, ssim_metric, auc_metric
from utils.seed import set_seed
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import re
import numpy as np

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("dcvae_epoch_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        raise FileNotFoundError("[ERROR] No checkpoints found in {}".format(checkpoint_dir))

    extract_epoch = lambda name: int(re.findall(r"epoch_(\d+)", name)[0])
    latest_file = max(checkpoint_files, key=extract_epoch)
    return os.path.join(checkpoint_dir, latest_file)

def save_test_visuals(case_idx, input_2d, recon_3d, original_3d, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    input_img = input_2d.detach().squeeze().cpu().numpy()
    recon_vol = recon_3d.detach().squeeze().cpu().numpy()
    gt_vol = original_3d.detach().squeeze().cpu().numpy()
    mid_slice = recon_vol.shape[0] // 2

    # Save PNG (cropped)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_img, cmap='viridis')
    axs[0].set_title("Input 2D")
    axs[1].imshow(recon_vol[mid_slice], cmap='viridis')
    axs[1].set_title("Recon z={}".format(mid_slice))
    axs[2].imshow(gt_vol[mid_slice], cmap='viridis')
    axs[2].set_title("GT z={}".format(mid_slice))
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"case_{case_idx}_comparison.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save RGB GIF of recon volume
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

    gif_path = os.path.join(output_dir, f"case_{case_idx}_reconstruction.gif")
    imageio.mimsave(gif_path, gif_frames, duration=0.15)

def main():
    with open("configs/config.yaml", 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(cfg['seed'])

    dataset = BraTS2021Dataset(cfg['paths']['dataset'], modality='flair')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = dCVAE(latent_dim=cfg['model']['latent_dim'],
                  input_channels=1,
                  output_shape=cfg['model']['output_3d_shape']).to(device)

    # Warm-up for dynamic FC layers
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 64, 64).to(device)
        dummy_time = torch.tensor([0.0]).to(device)
        model(dummy_input, dummy_time)

    # Load latest checkpoint
    checkpoint_path = get_latest_checkpoint(cfg['paths']['checkpoints'])
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()

    mse_scores, ssim_scores, auc_scores = [], [], []

    with torch.no_grad():
        for idx, volume in enumerate(tqdm(dataloader, desc="Testing")):
            volume = volume.to(device).float()
            input_slice = generate_noisy_2d_slice(volume[0]).float().to(device)
            t = torch.tensor([cfg['time_control']['prediction_times_months'][0]]).to(device)
            recon_volume, _, _, _ = model(input_slice.unsqueeze(0), t)

            mse_scores.append(mse_metric(recon_volume, volume))
            ssim_scores.append(ssim_metric(recon_volume, volume))
            auc_scores.append(auc_metric(recon_volume, volume))

            save_test_visuals(
                case_idx=idx,
                input_2d=input_slice,
                recon_3d=recon_volume,
                original_3d=volume,
                output_dir=os.path.join("results/test_outputs", f"case_{idx}")
            )

    print("\n=== Evaluation Metrics ===")
    print(f"MSE:  {sum(mse_scores)/len(mse_scores):.4f}")
    print(f"SSIM: {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"AUC:  {sum(auc_scores)/len(auc_scores):.4f}")

if __name__ == "__main__":
    main()
