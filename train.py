import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.dcvae import dCVAE
from utils.loss import dCVAE_loss
from utils.seed import set_seed
from data.dataloader import BraTS2021Dataset
from data.preprocess import generate_noisy_2d_slice
from utils.saver import save_checkpoint
from tqdm import tqdm
import yaml
import imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def save_epoch_visuals(epoch, input_2d, recon_3d, original_3d, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    input_img = input_2d.detach().squeeze().cpu().numpy()
    recon_vol = recon_3d.detach().squeeze().cpu().numpy()
    gt_vol = original_3d.detach().squeeze().cpu().numpy()
    mid_slice = recon_vol.shape[0] // 2

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_img, cmap='viridis')
    axs[0].set_title("Input 2D")
    axs[1].imshow(recon_vol[mid_slice], cmap='viridis')
    axs[1].set_title(f"Recon z={mid_slice}")
    axs[2].imshow(gt_vol[mid_slice], cmap='viridis')
    axs[2].set_title(f"GT z={mid_slice}")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"epoch_{epoch}_comparison.png")
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    gif_frames = []
    for z in range(recon_vol.shape[0]):
        fig, ax = plt.subplots(figsize=(3, 3))
        canvas = FigureCanvas(fig)
        ax.imshow(recon_vol[z], cmap='viridis')
        ax.axis('off')
        fig.tight_layout(pad=0)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        w, h = canvas.get_width_height()
        image = buf.reshape((h, w, 4))
        gif_frames.append(image)
        plt.close(fig)

    gif_path = os.path.join(output_dir, f"epoch_{epoch}_reconstruction.gif")
    imageio.mimsave(gif_path, gif_frames, duration=0.15)

def main():
    with open("configs/config.yaml", 'r') as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(cfg['seed'])

    dataset = BraTS2021Dataset(cfg['paths']['dataset'], modality='flair')
    dataloader = DataLoader(dataset,
                            batch_size=cfg['training']['batch_size'],
                            shuffle=True,
                            num_workers=cfg['training']['num_workers'])

    model = dCVAE(latent_dim=cfg['model']['latent_dim'],
                  input_channels=1,
                  output_shape=cfg['model']['output_3d_shape']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    loss_history = {"total": [], "recon": [], "kl": [], "tc": []}

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = total_recon_loss = total_kl_loss = total_tc_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")

        for i, volume in enumerate(progress_bar):
            volume = volume.to(device).float()
            noisy_2d = torch.stack([
                generate_noisy_2d_slice(vol[0]).float().to(device) for vol in volume
            ])

            t = torch.tensor([cfg['time_control']['prediction_times_months'][0]] * volume.size(0)).float().to(device)
            recon_volume, mu, logvar, z = model(noisy_2d, t)

            loss, recon_loss, kl_loss, tc_loss = dCVAE_loss(
                recon_volume, volume, mu, logvar, z, beta=1.0, lam=1.0
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_tc_loss += tc_loss.item()

            progress_bar.set_postfix(
                Loss=total_loss / (i + 1),
                Recon=total_recon_loss / (i + 1),
                KL=total_kl_loss / (i + 1),
                TC=total_tc_loss / (i + 1)
            )

            if i == 0:
                save_epoch_visuals(
                    epoch+1, noisy_2d[0].cpu(), recon_volume[0].cpu(), volume[0].cpu(),
                    output_dir=os.path.join("results/train_epoch_outputs", f"epoch_{epoch+1}")
                )

        loss_history["total"].append(total_loss)
        loss_history["recon"].append(total_recon_loss)
        loss_history["kl"].append(total_kl_loss)
        loss_history["tc"].append(total_tc_loss)

        os.makedirs(cfg['paths']['checkpoints'], exist_ok=True)
        checkpoint_path = os.path.join(cfg['paths']['checkpoints'], f"dcvae_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch+1, checkpoint_path)

    os.makedirs(cfg['paths']['results'], exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["total"], label="Total Loss")
    plt.plot(loss_history["recon"], label="Reconstruction Loss")
    plt.plot(loss_history["kl"], label="KL Loss")
    plt.plot(loss_history["tc"], label="TC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['paths']['results'], "training_loss_curves.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()
