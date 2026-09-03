#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:49:35 2026

@author: ire0279s
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround (use temporarily)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ot
import yaml
import numpy as np
from glob import glob
from typing import Tuple


# ============================================================
# 1) IO: read/load .bin (1 file = 1 image 64x64 float32, no header)
# ============================================================
def read_bin_image_64x64(path: str, order: str = "C", transpose: bool = False) -> np.ndarray:
    """
    Read a .bin with exactly 4096 float32 values (64x64), no header.
    Returns np.ndarray float32 [64,64].
    If transpose=True, returns arr.T (useful if you want (x,z)->(z,x) like your snippet).
    """
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != 64 * 64:
        raise ValueError(f"{path}: expected {64*64} float32 values, got {arr.size}.")
    img = arr.reshape((64, 64), order=order)
    return img.T if transpose else img

def load_folder_bins(folder: str, n_expected: int = 5000, order: str = "C", transpose: bool = False) -> np.ndarray:
    """
    Load all .bin files in a folder (1 file per image).
    Returns A: [N,64,64] float32.
    """
    paths = sorted(glob(os.path.join(folder, "*.bin")))
    if len(paths) == 0:
        raise ValueError(f"{folder}: no .bin files found.")
    if n_expected is not None and len(paths) != n_expected:
        raise ValueError(f"{folder}: expected {n_expected} .bin files, found {len(paths)}.")
    A = np.empty((len(paths), 64, 64), dtype=np.float32)
    for i, p in enumerate(paths):
        A[i] = read_bin_image_64x64(p, order=order, transpose=transpose)
    return A

# ============================================================
# 2) Preprocess (shared global stats recommended for MMD/GMMD)
#    Return [N,64,64] (sample = matrix)
# ============================================================
def preprocess_images_shared_stats_2d(
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A: [N,64,64] float32
    B: [M,64,64] float32
    Retorna A,B padronizados com média/std globais em A∪B.
    """
    if A.ndim != 3 or A.shape[1:] != (64, 64):
        raise ValueError(f"A must be [N,64,64], got {A.shape}")
    if B.ndim != 3 or B.shape[1:] != (64, 64):
        raise ValueError(f"B must be [M,64,64], got {B.shape}")

    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)

    all_vals = np.concatenate([A.ravel(), B.ravel()])
    mean = all_vals.mean()
    std = all_vals.std()
    std = max(std, 1e-8)

    A = (A - mean) / std
    B = (B - mean) / std
    
    return A, B



# ============================================================
# 2) End-to-end usage
# ============================================================
def main():
    
    with open('./00_configs/config_parameters.yaml') as file:
        parameter_data = yaml.safe_load(file)

    folder_A = parameter_data['paths']['dataset_path']
    folder_B = parameter_data['paths']['generated_seismic_images_path']

    # Se seu dado precisa de transpose (como seu snippet), coloque transpose=True aqui
    transpose = False

    A = load_folder_bins(folder_A, n_expected=5000, order="C", transpose=transpose)
    B = load_folder_bins(folder_B, n_expected=5000, order="C", transpose=transpose)

    # X, Y = preprocess_images_shared_stats_2d(A, B)
    
    x_spatial_flat = A.reshape(5000, -1)
    y_spatial_flat = B.reshape(5000, -1)
    print(np.shape(x_spatial_flat))
    
    w_dist = ot.sliced_wasserstein_distance(x_spatial_flat, y_spatial_flat, n_projections=5000)
    print(f"Sliced Wasserstein Distance - POT: {w_dist}")
    
    
if __name__ == "__main__":
    main()