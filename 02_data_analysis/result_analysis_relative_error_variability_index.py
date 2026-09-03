# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:28:04 2024

@author: chsba
"""

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt


# Configuration Parameters
with open('./00_configs/config_parameters.yaml') as file:
    parameter_data = yaml.safe_load(file)

dimension_x = parameter_data['images']['dimension_x']
dimension_z = parameter_data['images']['dimension_z']

spacing_x = 6.25
spacing_z = 6.25

scaling = 1.0/1000.0


print(' ')
print('Running...')

#
#   PROPOSED SEISMIC IMAGES
#

# Plotting the Generated Seismic Image Sample from VAE
generated_seismic_image_folder = parameter_data['paths']['generated_seismic_images_path']
generated_file_list = [os.path.join(generated_seismic_image_folder, f) for f in os.listdir(generated_seismic_image_folder) if f.endswith('.bin')]

# Create a list to store the N NumPy arrays
appended_seismic_images = []

for file_index in range(len(generated_file_list)):
    
    generated_seismic_image = np.fromfile(generated_file_list[file_index],dtype='float32')
    generated_seismic_image = generated_seismic_image.reshape(dimension_x,dimension_z)
    generated_seismic_image = generated_seismic_image.transpose()
    
    # appended_seismic_images.append(generated_seismic_image/np.max(np.abs(generated_seismic_image)))
    appended_seismic_images.append(generated_seismic_image)
    
    
# Stack the arrays along a new axis to create a 3D array (50x64x64)
stacked_seismic_images = np.stack(appended_seismic_images, axis=0)
stacked_seismic_images = 1.01 + stacked_seismic_images

# Calculate the element-wise variance across the 50 arrays
elementwise_seismic_image_mean = np.mean(stacked_seismic_images, axis=0)
elementwise_seismic_image_std = np.std(stacked_seismic_images, axis=0)


variability_index_proposed = 100.0 * (elementwise_seismic_image_std / elementwise_seismic_image_mean)


#
#   REFERENCE SEISMIC IMAGES
#

# Plotting the Reference Seismic Image Samples
reference_seismic_image_folder = parameter_data['paths']['training_dataset_path']
reference_file_list = [os.path.join(reference_seismic_image_folder, f) for f in os.listdir(reference_seismic_image_folder) if f.endswith('.bin')]
# print(len(reference_file_list))

# Create a list to store the N NumPy arrays
appended_reference_seismic_images = []

for file_index in range(len(reference_file_list)):
    
    reference_seismic_image = np.fromfile(reference_file_list[file_index],dtype='float32')
    reference_seismic_image = reference_seismic_image.reshape(dimension_x,dimension_z)
    reference_seismic_image = reference_seismic_image.transpose()
    
    # appended_reference_seismic_images.append(reference_seismic_image/np.max(np.abs(reference_seismic_image)))
    appended_reference_seismic_images.append(reference_seismic_image)


# Stack the arrays along a new axis to create a 3D array (50x64x64)
reference_stacked_seismic_images = np.stack(appended_reference_seismic_images, axis=0)
reference_stacked_seismic_images = 1.01 + reference_stacked_seismic_images


# Calculate the element-wise variance across the 50 arrays
elementwise_seismic_image_mean = np.mean(reference_stacked_seismic_images, axis=0)
elementwise_seismic_image_std = np.std(reference_stacked_seismic_images, axis=0)


variability_index_reference = 100.0 * (elementwise_seismic_image_std / elementwise_seismic_image_mean)


print('Printing results...')

# Coordenadas do centro do "X"
x_center = 16*spacing_x*scaling
y_center = 32*spacing_z*scaling

x_center2 = 32*spacing_x*scaling
y_center2 = 32*spacing_z*scaling

x_center3 = 32*spacing_x*scaling
y_center3 = 16*spacing_z*scaling

x_center4 = 12*spacing_x*scaling
y_center4 = 48*spacing_z*scaling

### Proposed variability index
auxiliar_path = parameter_data['paths']['seismic_image_results']
variability_index_path = f"{auxiliar_path}/variability_index_proposed.png"

# Scaling the axis to the original dimension
xmin = (0 * spacing_x) * scaling
xmax = (dimension_x * spacing_x) * scaling
zmin = (0 * spacing_z) * scaling
zmax = (dimension_z * spacing_z) * scaling

# setting xtick on top
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labeltop'] = True

plt.rc('xtick', labelsize=14)     
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'font.size': 14})
    
# Setting figure
fig, ax = plt.subplots(figsize=(15,4))
ax.xaxis.set_label_position('top') # moving extension to the top
edit_fig = ax.imshow(variability_index_proposed, cmap='Greys', vmin=0.0, vmax=10.0, interpolation='bicubic', extent=[xmin,xmax,zmax,zmin], aspect=1.6)
cbar = fig.colorbar(edit_fig, ax=ax, fraction=0.09, pad=0.02, label='Variability Index (%)', aspect=15)
plt.xlabel('Extension (Km)', fontsize=14)
plt.ylabel('Depth (Km)', fontsize=14)

plt.scatter(x_center, y_center, marker='x', color='red', s=100)
plt.scatter(x_center2, y_center2, marker='x', color='red', s=100)
plt.scatter(x_center3, y_center3, marker='x', color='red', s=100)
plt.scatter(x_center4, y_center4, marker='x', color='red', s=100)

ax.text(0.04,0.025, "(b)", size=12, rotation=0.,
          ha="center", va="center",
          bbox=dict(boxstyle="round", facecolor='white'))

ax.text(x_center,y_center+0.02, "Pixel 2", size=7, rotation=0., ha="right", va="top")
ax.text(x_center2,y_center2+0.02, "Pixel 3", size=7, rotation=0., ha="left", va="top")
ax.text(x_center3,y_center3+0.02, "Pixel 1", size=7, rotation=0., ha="left", va="top")
ax.text(x_center4,y_center4+0.02, "Pixel 4", size=7, rotation=0., ha="left", va="top")

plt.savefig(variability_index_path, dpi=300, bbox_inches='tight');


### Mean reference seismic index
auxiliar_path = parameter_data['paths']['seismic_image_results']
confidence_index_path = f"{auxiliar_path}/variability_index_reference.png"

# Scaling the axis to the original dimension
xmin = (0 * spacing_x) * scaling
xmax = (dimension_x * spacing_x) * scaling
zmin = (0 * spacing_z) * scaling
zmax = (dimension_z * spacing_z) * scaling

# setting xtick on top
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labeltop'] = True

plt.rc('xtick', labelsize=14)     
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'font.size': 14})
    
# Setting figure
fig, ax = plt.subplots(figsize=(15,4))
ax.xaxis.set_label_position('top') # moving extension to the top
edit_fig = ax.imshow(variability_index_reference, cmap='Greys', vmin=0.0, vmax=10.0, interpolation='bicubic', extent=[xmin,xmax,zmax,zmin], aspect=1.6)
cbar = fig.colorbar(edit_fig, ax=ax, fraction=0.09, pad=0.02, label='Variability Index (%)', aspect=15)
plt.xlabel('Extension (Km)', fontsize=14)
plt.ylabel('Depth (Km)', fontsize=14)

plt.scatter(x_center, y_center, marker='x', color='red', s=100)
plt.scatter(x_center2, y_center2, marker='x', color='red', s=100)
plt.scatter(x_center3, y_center3, marker='x', color='red', s=100)
plt.scatter(x_center4, y_center4, marker='x', color='red', s=100)

ax.text(0.04,0.025, "(a)", size=12, rotation=0.,
          ha="center", va="center",
          bbox=dict(boxstyle="round", facecolor='white'))

ax.text(x_center,y_center+0.02, "Pixel 2", size=7, rotation=0., ha="right", va="top")
ax.text(x_center2,y_center2+0.02, "Pixel 3", size=7, rotation=0., ha="left", va="top")
ax.text(x_center3,y_center3+0.02, "Pixel 1", size=7, rotation=0., ha="left", va="top")
ax.text(x_center4,y_center4+0.02, "Pixel 4", size=7, rotation=0., ha="left", va="top")

plt.savefig(confidence_index_path, dpi=300, bbox_inches='tight');   



print(' ')
print('Finished...')
