import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        
        return self.fc_mu(h5), self.fc_logvar(h5)


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        # Encoder와 Decoder 정의
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim, input_dim=input_dim)
        self.prior = CompositePrior(hidden_dim=hidden_dim, latent_dim=latent_dim, input_dim=input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1.0, dropout_rate=0.5, calculate_loss=True):
        """
        Forward pass for the VAE model.

        Args:
            user_ratings (torch.Tensor): Input tensor of user ratings.
            beta (float): Weight for KL divergence in the ELBO loss.
            gamma (float): Scaling factor for KL divergence.
            dropout_rate (float): Dropout rate for encoder.
            calculate_loss (bool): Whether to calculate and return the loss.

        Returns:
            If calculate_loss is True:
                Tuple[Tuple[float, float], float]: MLL and KLD values along with the negative ELBO loss.
            Else:
                torch.Tensor: Predicted ratings.
        """
        # Encoding
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)

        # Decoding
        x_pred = self.decoder(z)

        if calculate_loss:
            # KL divergence weight
            kl_weight = gamma * user_ratings.sum(dim=-1) if gamma else beta

            # Marginal log likelihood (MLL)
            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()

            # KL divergence (KLD)
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()

            # Negative ELBO loss
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo
        else:
            return x_pred

    def update_prior(self):
        """Update the prior with the current encoder's parameters."""
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))