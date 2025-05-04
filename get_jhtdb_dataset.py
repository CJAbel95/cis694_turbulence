#
# Updated to extract physically meaningful velocity slices for DL training
#
# This represents the portion of Javad's code that downloads images from
# the JHTDB using the givernylocal library.
#

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from givernylocal.turbulence_gizmos.basic_gizmos import write_interpolation_tsv_file
from torch.utils.data import Dataset, DataLoader
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData
import os
import time

# === PARAMETERS ===
save_tsv_file = False
output_tsv_file = 'turbulence_out_p'
# auth_token = 'edu.jhu.pha.turbulence.testing-201406'
# auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
auth_token = 'edu.csuohio.c.j.abel-3584bba4'
dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'
# save_img_dir = './training_slices'
save_img_dir = './Isotropic turbulence/training_slices2'
variable = 'pressure'
spatial_method = 'lag6'
temporal_method = 'none'
spatial_operator = 'field'

# Create folder to save training data images
os.makedirs(save_img_dir, exist_ok=True)

# === Instantiate Dataset ===
dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)

# === Generate 2D u-velocity slices over time ===
nx = ny = 128
x_points = np.linspace(0.0, 2 * np.pi, nx)
y_points = np.linspace(0.0, 2 * np.pi, ny)
z = np.pi  # midplane at z = π

T_start = 0.004
T_end = 5.0
T_delta = 0.002
T_list = np.arange(T_start, T_end + T_delta, T_delta)
# T_list = np.array([0.1])

slices = []

for i, t in enumerate(T_list):
    getdata_success =  False
    while not getdata_success:
        time.sleep(10) # Wait 10s at the start of each query
        print(f"Querying time: {t:.3f}...")
        points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, [z], indexing='ij')], dtype=np.float64).T
        try:
            result = getData(dataset, variable, t, temporal_method, spatial_method, spatial_operator, points)
            # ux_field = np.array(result[0])[:, 0].reshape((nx, ny))
            # uy_field = np.array(result[0])[:, 1].reshape((nx, ny))
            p_field = np.array(result[0])[:, 0].reshape((nx, ny))
            # slices.append(ux_field)

            # Save ux to image
            plt.imsave(f"{save_img_dir}/p/p_t{t:.3f}.png", p_field, cmap='seismic', vmin=-3, vmax=3)
            # Save ux to image
            # plt.imsave(f"{save_img_dir}/uy/uy_t{t:.3f}.png", uy_field, cmap='seismic', vmin=-3, vmax=3)

            # Save to TSV file
            if (save_tsv_file) and (i == 1):
                print("Saving data at t = {0:.3f} to {1}_t{1}.tsv".format(t, output_tsv_file, i))
                write_interpolation_tsv_file(dataset, points, result, "{0}_t{1}".format(output_tsv_file, i))

            # If the save to image was successful, go on to the next tuple of i, t
            getdata_success = True
        except Exception as e:
            print(f"Warning: Failed at t={t:.3f} due to {e}. Moving onto the next query after 10s delay.")

# slices = np.array(slices)
# print("Shape of dataset:", slices.shape)

# === Normalize ===
# mean = slices.mean()
# std = slices.std()
# slices = (slices - mean) / std


