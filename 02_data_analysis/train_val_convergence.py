# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:53:35 2024

@author: chsba
"""

import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration Parameters
with open('./00_configs/config_parameters.yaml') as file:
    parameter_data = yaml.safe_load(file)


# Caminho para o arquivo de saída gerado anteriormente
convergence_file_path = parameter_data['paths']['convergence_file_path']

# Ler o arquivo CSV
data = pd.read_csv(convergence_file_path, sep=',')

# Normalizando os dados para a escala logarítmica
normalized_convergence_data = data['Train_Loss'] / (np.abs(np.max(data['Train_Loss'])))
normalized_convergence_data = np.log(normalized_convergence_data)

normalized_reconstruction_loss_data = data['Train_Reconstruction_Loss'] / (np.abs(np.max(data['Train_Loss'])))
normalized_reconstruction_loss_data = np.log(normalized_reconstruction_loss_data)

normalized_kl_divergence_data = data['Train_KL_Divergence_c'] / (np.abs(np.max(data['Train_Loss'])))
normalized_kl_divergence_data = np.log(normalized_kl_divergence_data)

normalized_kl_z_divergence_data = data['Train_KL_Divergence_z'] / (np.abs(np.max(data['Train_Loss'])))
normalized_kl_z_divergence_data = np.log(normalized_kl_z_divergence_data)

val_normalized_convergence_data = data['Val_Loss'] / (np.abs(np.max(data['Val_Loss'])))
val_normalized_convergence_data = np.log(val_normalized_convergence_data)

val_normalized_reconstruction_loss_data = data['Val_Reconstruction_Loss'] / (np.abs(np.max(data['Val_Loss'])))
val_normalized_reconstruction_loss_data = np.log(val_normalized_reconstruction_loss_data)

val_normalized_kl_divergence_data = data['Val_KL_Divergence_c'] / (np.abs(np.max(data['Val_Loss'])))
val_normalized_kl_divergence_data = np.log(val_normalized_kl_divergence_data)

val_normalized_kl_z_divergence_data = data['Val_KL_Divergence_z'] / (np.abs(np.max(data['Val_Loss'])))
val_normalized_kl_z_divergence_data = np.log(val_normalized_kl_z_divergence_data)

# Plotar as três curvas
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(data['Epoch'], normalized_convergence_data, label='Train Loss Function', color='b', linestyle='-')
ax.plot(data['Epoch'], normalized_reconstruction_loss_data, label='Train Reconstruction Error', color='g', linestyle='--')
ax.plot(data['Epoch'], normalized_kl_divergence_data, label='Train  KL Divergence c', color='r', linestyle=':')
ax.plot(data['Epoch'], normalized_kl_z_divergence_data, label='Train KL Divergence z', color='purple', linestyle='dashed')

ax.plot(data['Epoch'], val_normalized_convergence_data, label='Validation Loss Function', color='cornflowerblue', linestyle='-')
ax.plot(data['Epoch'], val_normalized_reconstruction_loss_data, label='Validation Reconstruction Error', color='lime', linestyle='--')
ax.plot(data['Epoch'], val_normalized_kl_divergence_data, label='Validation  KL Divergence c', color='lightcoral', linestyle=':')
ax.plot(data['Epoch'], val_normalized_kl_z_divergence_data, label='Validation KL Divergence z', color='violet', linestyle='dashed')

# Ajustar o eixo X para ficar abaixo do gráfico
# ax.spines['bottom'].set_position(('outward', 10))  # move o eixo X para fora
ax.xaxis.set_ticks_position('bottom')
# ax.spines['top'].set_visible(False)  # remove a linha superior do gráfico
plt.rc('xtick', labelsize=14)     
plt.rc('ytick', labelsize=14)
# plt.rcParams.update({'font.size': 14})
# Adicionar rótulos e legenda
plt.xlabel('Epoches', fontsize=16)
plt.ylabel('Training/Validation Loss Function (Log Scale)', fontsize=16)
plt.legend()


# Mostrar o gráfico
convergence_plot = parameter_data['paths']['seismic_image_results']
plt.savefig(f"{convergence_plot}train_val_convergence.png", dpi=300, bbox_inches='tight')
plt.show()