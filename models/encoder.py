import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, input_channels=1):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Temporarily initialize with a safe flat_dim
        self.fc_mu = nn.Identity()
        self.fc_logvar = nn.Identity()
        self._fc_init_done = False

    def _initialize_fc(self, flat_dim):
        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
        self._fc_init_done = True
        print(f"[DEBUG] FC layers initialized with flat_dim: {flat_dim}")

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)

        # Dynamically initialize fc layers during first forward
        if not self._fc_init_done:
            self._initialize_fc(x.shape[1])

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
