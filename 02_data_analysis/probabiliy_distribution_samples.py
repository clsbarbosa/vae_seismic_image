# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:42:09 2025

@author: chsba
"""

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance


# Configuration Parameters
with open('./00_configs/config_parameters.yaml') as file:
    parameter_data = yaml.safe_load(file)

dimension_x = parameter_data['images']['dimension_x']
dimension_z = parameter_data['images']['dimension_z']


print(' ')
print('Running...')

#
#   PROPOSED SEISMIC IMAGES
#
# Plotting the Generated Seismic Image Sample from VAE
generated_seismic_image_folder = parameter_data['paths']['generated_seismic_images_path']
generated_file_list = [os.path.join(generated_seismic_image_folder, f) for f in os.listdir(generated_seismic_image_folder) if f.endswith('.bin')]


#
#   REFERENCE SEISMIC IMAGES
#
# Plotting the Reference Seismic Image Samples
reference_seismic_image_folder = parameter_data['paths']['full_dataset_path']
reference_file_list = [os.path.join(reference_seismic_image_folder, f) for f in os.listdir(reference_seismic_image_folder) if f.endswith('.bin')]


# Create a list to store the N NumPy arrays
appended_reference_seismic_images = []
for file_index in range(len(reference_file_list)):
    
    reference_seismic_image = np.fromfile(reference_file_list[file_index],dtype='float32')
    reference_seismic_image = reference_seismic_image.reshape(dimension_x,dimension_z)
    reference_seismic_image = reference_seismic_image.transpose()
    
    appended_reference_seismic_images.append(reference_seismic_image)
    

appended_generated_seismic_images = []
for file_index in range(len(generated_file_list)):
    
    generated_seismic_image = np.fromfile(generated_file_list[file_index],dtype='float32')
    generated_seismic_image = generated_seismic_image.reshape(dimension_x,dimension_z)
    generated_seismic_image = generated_seismic_image.transpose()
    
    appended_generated_seismic_images.append(generated_seismic_image)
    
    

# Stack the arrays along a new axis to create a 3D array (5000x64x64)
stacked_reference_seismic_images = np.stack(appended_reference_seismic_images, axis=0)
stacked_generated_seismic_images = np.stack(appended_generated_seismic_images, axis=0)



params_list = [
    {
        "point_x": 32,
        "point_z": 16,
        "controlX1": -0.02,
        "controlX2": 0.12,
        "controlY": 50
    },
    {
        "point_x": 16,
        "point_z": 32,
        "controlX1": -0.13,
        "controlX2": -0.06,
        "controlY": 150
    },
    {
        "point_x": 32,
        "point_z": 32,
        "controlX1": -0.025,
        "controlX2": 0.15,
        "controlY": 50
    },
    {
        "point_x": 12,
        "point_z": 48,
        "controlX1": 0.335,
        "controlX2": 0.53,
        "controlY": 30
    }
]


output_metrics_path = parameter_data['paths']['metrics_results_path']

with open(output_metrics_path, "w") as f:

    for params in params_list:
    
        point_x = params["point_x"]
        point_z = params["point_z"]
    
        controlX1 = params["controlX1"]
        controlX2 = params["controlX2"]
        controlY  = params["controlY"]
    
        print("Processing Pixel: ", point_x, point_z)
        f.write(f"Processing Pixel: {point_x} {point_z}\n")

        
        reference_element_wise_distribution = np.zeros(len(reference_file_list))
        for file_index in range(len(reference_file_list)):
            reference_element_wise_distribution[file_index] = stacked_reference_seismic_images[file_index, point_x, point_z]
            
        generated_element_wise_distribution = np.zeros(len(generated_file_list))
        for file_index in range(len(generated_file_list)):
            generated_element_wise_distribution[file_index] = stacked_generated_seismic_images[file_index, point_x, point_z]
        
        
        wasser_dist_norm = wasserstein_distance(reference_element_wise_distribution, generated_element_wise_distribution)
        print("Wasserstein Distance: ", wasser_dist_norm)
        f.write(f"Pixel-wise Wasserstein Distance: {wasser_dist_norm}\n")

        plt.figure(figsize=(5, 3))
        plt.xlabel('Dataset Samples')
        plt.ylabel('Probability Density')
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        colors = ['Dataset Samples']
        plt.hist(reference_element_wise_distribution, bins=100, color='blue', edgecolor='black', density=True, histtype='stepfilled', label='Dataset Samples', linewidth=1.5)
        ax.set_xlim([controlX1, controlX2])
        ax.set_ylim([0.0, controlY])
        plt.legend(fontsize=12, loc='upper left')
        plt.savefig(f"02_distributions/marginal_density_dataset_point_{point_x}x{point_z}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(5, 3))
        plt.xlabel('mVAE Samples')
        plt.ylabel('Probability Density')
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        colors = ['mVAE Samples']
        plt.hist(generated_element_wise_distribution, bins=100, color='red', edgecolor='black', density=True, histtype='stepfilled', label='mVAE Samples', linewidth=1.5)
        ax.set_xlim([controlX1, controlX2])
        ax.set_ylim([0.0, controlY])
        plt.legend(fontsize=12, loc='upper left')
        plt.savefig(f"02_distributions/marginal_density_mvae_point_{point_x}x{point_z}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(5, 3))
        plt.xlabel('Dataset Samples   |   mVAE Samples')
        plt.ylabel('Probability Density')
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        colors = ['Dataset Samples', 'mVAE Samples']
        plt.hist(generated_element_wise_distribution, bins=100, color='red', edgecolor='black', density=True, histtype='stepfilled', label='mVAE Samples', linewidth=1.5)
        ax.set_xlim([controlX1, controlX2])
        ax.set_ylim([0.0, controlY])
        plt.hist(reference_element_wise_distribution, bins=100, color='blue', alpha=1.0, edgecolor='black', density=True, histtype='stepfilled', label='Dataset Samples', linewidth=1.5)
        plt.legend(fontsize=12, loc='upper left')
        plt.savefig(f"02_distributions/marginal_density_point_refxmvae_{point_x}x{point_z}.png", dpi=300, bbox_inches='tight')
        plt.show()