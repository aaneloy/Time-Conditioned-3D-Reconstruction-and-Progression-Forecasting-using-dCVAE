import torch
import torch.nn as nn
from models.encoder import Encoder, reparameterize
from models.decoder3d import Decoder3D

class dCVAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=1, output_shape=(64, 64, 64)):
        super(dCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, input_channels)
        self.decoder = Decoder3D(latent_dim, output_shape)

    def forward(self, x, t):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstructed_volume = self.decoder(z, t)
        return reconstructed_volume, mu, logvar, z
