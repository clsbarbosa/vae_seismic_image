# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:51:20 2024

@author: chsba
"""

import yaml
import torch

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.distributions import Normal
from torch import nn


with open('./00_configs/config_parameters.yaml') as file:
    parameter_data = yaml.safe_load(file)


# Construção do autoencoder
# Define the dimensions and architecture of the encoder
input_encoder_dim = 4096
output_encoder_dim = 64

# Define the dimensions and architecture of the decoder
input_decoder_dim = 64
output_decoder_dim = 4096

# Define the latent space dimension
latent_space_dim = 64
discrete_latent_space_dim = 64

# Define the activation function
activation =  nn.ELU()

# Define the dropout factor
dropout_factor = 0.1


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
# Define the autoencoder class
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        
        # Model parameters
        self.num_categories = discrete_latent_space_dim
        self.latent_dim = latent_space_dim
        
        # Trainable prior parameters for each category (shared for all z's)
        self.prior_mu = nn.Parameter(torch.zeros(self.num_categories, self.latent_dim), requires_grad=True)
        self.prior_logvar = nn.Parameter(torch.zeros(self.num_categories, self.latent_dim), requires_grad=True)
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16,  kernel_size=7, stride=2, padding_mode='reflect', padding=3),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.Conv2d(16, 32,  kernel_size=7, stride=2, padding_mode='reflect', padding=3),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.Conv2d(32, 64,  kernel_size=7, stride=2, padding_mode='reflect', padding=3),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.Flatten()
        )
        
        self.fc_mu       = nn.Linear(64*8*8 + self.num_categories, self.latent_dim)
        self.fc_logvar   = nn.Linear(64*8*8 + self.num_categories, self.latent_dim)
        self.fc_discrete = nn.Linear(64*8*8, self.num_categories)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_dim+self.num_categories, 64*8*8),
            Reshape(-1, 64, 8, 8),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.ConvTranspose2d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            activation,
            nn.Dropout2d(dropout_factor),
            nn.ConvTranspose2d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x, temperature=1.0):
        h = self.encoder(x)
        logits_c = self.fc_discrete(h)
        c = self.reparameterize_discrete(logits_c, temperature)
        
        # Continuous variable z conditioned on c
        z = torch.cat([h, c], dim=1)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        
        return mu, logvar, logits_c, c
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)                                             # or std = logvar like https://github.com/t-m-d-k/Pytorch-Series/blob/main/CustomVAE-Final.ipynb
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def reparameterize_discrete(self, logits_c, temperature=1.0):
        
        y_soft = F.gumbel_softmax(logits_c, tau=temperature, hard=True)        # Gumbel-Softmax for differentiable sampling
        
        return y_soft
    
    def compute_entropy(self, logits_c):
        probs = F.softmax(logits_c, dim=-1)
        log_probs = F.log_softmax(logits_c, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def kl_divergence_gaussian(self, mu1, logvar1, mu2, logvar2):
        return 0.5 * torch.sum(
            logvar2 - logvar1 +
            (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1,
            dim=1
        )

    def loss_function(self, x_recon, x, mu, logvar, logits_c, beta):
        batch_size = x.size(0)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size

        # Categorical KL
        probs_c = F.softmax(logits_c, dim=-1)
        kl_c = -self.compute_entropy(logits_c).mean() + torch.log(torch.tensor(self.num_categories, dtype=torch.float32, device=x.device))
        
        # KL between q(z|x,c) and p(z|c)
        kl_z = 0
        for k in range(self.num_categories):
            prior_mu_k = self.prior_mu[k].unsqueeze(0).expand(batch_size, -1)
            prior_logvar_k = self.prior_logvar[k].unsqueeze(0).expand(batch_size, -1)
            kl_k = self.kl_divergence_gaussian(mu, logvar, prior_mu_k, prior_logvar_k)
            kl_z += probs_c[:, k] * kl_k
        kl_z = kl_z.mean()

        total_loss = recon_loss + beta * (kl_c + kl_z)
        return total_loss, recon_loss, kl_c, kl_z
    
    def sample(self, num_samples, device, temperature=0.01):
        c_logits = torch.zeros(num_samples, self.num_categories, device=device)
        c = self.reparameterize_discrete(c_logits, temperature)

        z = torch.zeros(num_samples, self.latent_dim, device=device)
        category_indices = torch.argmax(c, dim=1)

        for k in range(self.num_categories):
            idx_k = (category_indices == k).nonzero(as_tuple=True)[0]
            if idx_k.numel() > 0:
                prior_dist = Normal(self.prior_mu[k].to(device), torch.exp(0.5 * self.prior_logvar[k]).to(device))
                z_k = prior_dist.sample((idx_k.numel(),)).to(device)
                z[idx_k] = z_k  # Assign z_k only to samples in category k

        x_gen = self.decoder(torch.cat([z, c], dim=1))
        return x_gen
    
    def forward(self, x, temperature=1.0):
        mu, logvar, logits_c, c = self.encode(x, temperature)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(torch.cat([z, c], dim=1))
        return decoded, mu, logvar, logits_c, c
    
    
# Instanciar o VAE
vae = VariationalAutoEncoder()

# Mover o VAE para a GPU, se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)


network_path = parameter_data['paths']['trained_vae_network_path']
state_dict = torch.load(network_path)
vae.load_state_dict(state_dict)


################################################
### RANDOM SAMPLE FROM A NORMAL DISTRIBUTION ###
################################################

numberOfSeismicImages = parameter_data['generator']['numberOfSeismicImages']
# numberOfSeismicImages = 5
auxiliar_path = parameter_data['paths']['generated_seismic_images_path']


vae.eval()
with torch.no_grad():
    samples = vae.sample(num_samples=numberOfSeismicImages, device=device, temperature=1.0)


for images_id in tqdm(range(numberOfSeismicImages), desc="Number of Seismic Images", unit="images"):
    
    generated_seismic_image = np.array(samples[images_id].to(torch.device('cpu'))).reshape((64, 64))
    
    filename = f"{auxiliar_path}generated_sample_{str(images_id+1).zfill(5)}.bin"
    
    generated_seismic_image.tofile(filename)