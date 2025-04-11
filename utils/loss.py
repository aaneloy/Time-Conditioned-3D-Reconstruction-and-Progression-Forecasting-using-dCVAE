import torch
import torch.nn.functional as F

def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def total_correlation(z, mu, logvar):
    # Estimate total correlation using minibatch samples
    # z: (batch_size, latent_dim)
    # mu, logvar: same shape
    batch_size, latent_dim = z.size()

    log_qz_prob = -0.5 * ((z.unsqueeze(1) - mu.unsqueeze(0)) ** 2 / logvar.exp().unsqueeze(0))
    log_qz_prob = log_qz_prob.sum(2)  # Sum over latent_dim → shape (B, B)

    log_qz = torch.logsumexp(log_qz_prob, dim=1) - torch.log(torch.tensor(batch_size, dtype=torch.float32, device=z.device))
    log_qz_product = torch.sum(torch.logsumexp(log_qz_prob, dim=0), dim=0) - latent_dim * torch.log(torch.tensor(batch_size, dtype=torch.float32, device=z.device))

    tc = (log_qz - log_qz_product).mean()
    return tc

def dCVAE_loss(recon_x, x, mu, logvar, z, beta=1.0, lam=1.0):
    recon_loss = reconstruction_loss(recon_x, x)
    kl_loss = kl_divergence(mu, logvar)
    tc_loss = total_correlation(z, mu, logvar)
    total_loss = recon_loss + beta * kl_loss + lam * tc_loss
    return total_loss, recon_loss, kl_loss, tc_loss
