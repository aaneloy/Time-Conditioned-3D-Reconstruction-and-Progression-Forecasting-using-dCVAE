import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder3D(nn.Module):
    def __init__(self, latent_dim=128, output_shape=(64, 64, 64), time_embed_dim=16):
        super(Decoder3D, self).__init__()

        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.output_shape = output_shape  # (D, H, W)

        # Time embedding layer
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        # Fully connected layer combines latent z and time embedding
        self.fc = nn.Linear(latent_dim + time_embed_dim, 512 * 4 * 4 * 4)

        # 3D convolutional transpose (deconv) layers to reconstruct 3D volume
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # (8,8,8)
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # (16,16,16)
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),   # (32,32,32)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),     # (64,64,64)
            nn.Sigmoid()
        )

    def forward(self, z, t):
        # Embed the scalar time into a learned embedding
        t = t.view(-1, 1).float()  # Ensure correct shape
        t_embed = self.time_embedding(t)

        # Concatenate latent variables with time embedding
        z_combined = torch.cat([z, t_embed], dim=1)

        # Pass through fully connected layer
        x = self.fc(z_combined)
        x = x.view(-1, 512, 4, 4, 4)  # Reshape into 3D volume format

        # Pass through deconvolutional layers to reconstruct 3D image
        volume = self.deconv_layers(x)

        # Output shape: [Batch, 1, D, H, W]
        return volume
