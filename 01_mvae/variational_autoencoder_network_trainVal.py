# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:38:14 2024

@author: chsba
"""

import os
import random
import yaml
import shutil

import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

print(torch.cuda.is_available())


seed_value = 100  # Arbitrary fixed integer; can be parameterized via YAML
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True  # Further enforces determinism
    torch.backends.cudnn.benchmark = False


# Assuming each binary file contains a 2D array of shape (64, 64)
def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the binary file and convert it to a NumPy array
        data = np.fromfile(f, dtype=np.float32).reshape((1, 64, 64))
        # data = data/np.abs(np.max(data))
    return data

class CustomDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = torch.from_numpy(read_binary_file(file_path))  # Assuming you have a function to read the binary file
        return data
    

print(' ')
print('STARTING CONDITIONAL VARIATIONAL AUTOENCODER...')
print(' ')

with open('./00_configs/config_parameters.yaml') as file:
    parameter_data = yaml.safe_load(file)
    
print('Hyperparameters:')
print(parameter_data['hyperparameters'])


print(' ')
print('Reading dataset...')


# Build the DataLoader
data_folder = parameter_data['paths']['dataset_path']
file_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.bin')]

random.shuffle(file_list)

# Split the file_list into training and validation subsets (80/20)
split_ratio = 0.8
train_size = int(len(file_list) * split_ratio)
train_files = file_list[:train_size]
val_files = file_list[train_size:]

# Create custom dataset objects for training and validation
train_dataset = CustomDataset(train_files)
val_dataset = CustomDataset(val_files)


train_folder = parameter_data['paths']['training_dataset_path']
val_folder = parameter_data['paths']['validation_dataset_path']

# Clear existing files in train and val folders if any
for folder in [train_folder, val_folder]:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Copy the files to the respective folders
for file_path in train_files:
    shutil.copy(file_path, train_folder)

for file_path in val_files:
    shutil.copy(file_path, val_folder)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


# Create data loaders for training and validation
batch_size = parameter_data['hyperparameters']['batch_size']  # You can adjust the batch size according to your requirements
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


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
        
        y_soft = F.gumbel_softmax(logits_c, tau=temperature, hard=False)        # Gumbel-Softmax for differentiable sampling
        
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
vae.to(device)

# Definir o otimizador, taxa de aprendizado
optimizer = optim.Adam(vae.parameters(), lr=parameter_data['hyperparameters']['learning_rate'])

vae.train()

# Configurar o número de épocas e tamanho do lote
epochs = parameter_data['hyperparameters']['epochs']
beta   = parameter_data['hyperparameters']['beta']

# Temperature scheduling parameters
tau_start = 1.0
tau_decay_rate = 0.995
tau_min = 0.1


print(' ')
print('Training variational autoencoder network...')

# Specify the file path
convergence_file_path = parameter_data['paths']['convergence_file_path']
with open(convergence_file_path, 'w') as file_output_path:
    
    # Cabeçalho para as colunas no arquivo
    file_output_path.write('Epoch,Train_Loss,Train_Reconstruction_Loss,Train_KL_Divergence_c,Train_KL_Divergence_z,Val_Loss,Val_Reconstruction_Loss,Val_KL_Divergence_c,Val_KL_Divergence_z\n')
    # Loop de treinamento
    for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"): 
        
        total_train_loss = 0.
        total_train_recon = 0.
        total_train_kl_c = 0.
        total_train_kl_z = 0.
        
        # Compute temperature (tau) with exponential decay
        tau = max(tau_min, tau_start * (tau_decay_rate ** epoch))
        
        # Training phase
        vae.train()
        for batch_data in train_dataloader:
            
            inputs = batch_data.to(device)
            
            optimizer.zero_grad()
            
            outputs, mu, logvar, logits_c, c = vae(inputs, temperature=1.0)
            
            loss, recon_loss, kl_c, kl_z = vae.loss_function(outputs, inputs, mu, logvar, logits_c, beta)
            
            loss.backward()
            
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_recon += recon_loss.item()
            total_train_kl_c += kl_c.item()
            total_train_kl_z += kl_z.item()
            
        # Compute training averages
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_recon = total_train_recon / len(train_dataloader)
        avg_train_kl_c = total_train_kl_c / len(train_dataloader)
        avg_train_kl_z = total_train_kl_z / len(train_dataloader)
        
        # Validation phase
        vae.eval()
        total_val_loss = 0.
        total_val_recon = 0.
        total_val_kl_c = 0.
        total_val_kl_z = 0.
        with torch.no_grad():
            for batch_data in val_dataloader:
                
                inputs = batch_data.to(device)
                
                outputs, mu, logvar, logits_c, c = vae(inputs, temperature=1.0)
                
                loss, recon_loss, kl_c, kl_z = vae.loss_function(outputs, inputs, mu, logvar, logits_c, beta)
                
                total_val_loss += loss.item()
                total_val_recon += recon_loss.item()
                total_val_kl_c += kl_c.item()
                total_val_kl_z += kl_z.item()
                
        # Compute validation averages
        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.
        avg_val_recon = total_val_recon / len(val_dataloader) if len(val_dataloader) > 0 else 0.
        avg_val_kl_c = total_val_kl_c / len(val_dataloader) if len(val_dataloader) > 0 else 0.
        avg_val_kl_z = total_val_kl_z / len(val_dataloader) if len(val_dataloader) > 0 else 0.
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Recon: {avg_train_recon:.4f}, Train KL_c: {avg_train_kl_c:.4f}, Train KL_z: {avg_train_kl_z:.4f}, Val Loss: {avg_val_loss:.4f}, Val Recon: {avg_val_recon:.4f}, Val KL_c: {avg_val_kl_c:.4f}, Val KL_z: {avg_val_kl_z:.4f}')
        file_output_path.write(f'{epoch + 1},{avg_train_loss:.8f},{avg_train_recon:.8f},{avg_train_kl_c:.8f},{avg_train_kl_z:.8f},{avg_val_loss:.8f},{avg_val_recon:.8f},{avg_val_kl_c:.8f},{avg_val_kl_z:.8f}\n')


network_path = parameter_data['paths']['trained_vae_network_path']
torch.save(vae.state_dict(), network_path)