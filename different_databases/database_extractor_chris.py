#
# Filename: database_extractor_chris.py
#
# Description: Extract One Variable from JHTDB Dataset.
#               Modified to include saving extracted data to csv file.
#
# Revision History
# ----------------
#   04/15/2025 -- Original
#
# ------------------------------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

# === PARAMETERS ===
auth_token = 'edu.jhu.pha.turbulence.testing-201406'
# auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
# dataset_title = 'isotropic1024fine'
dataset_title = 'channel5200'
#dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'
save_img_dir = './'
variable = 'velocity'
spatial_method = 'lag4'
temporal_method = 'none' # 'none' or 'pchip'
spatial_operator = 'field'

# Create folder to save training data images
os.makedirs(save_img_dir, exist_ok=True)

# === Instantiate Dataset ===
dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)

# === Generate 2D u-velocity slices over time ===
nx = ny = 16
x_points = np.linspace(0, 8*np.pi , nx)
y_points = np.linspace(-1, 1, ny)
z = 1.5*np.pi # midplane at z = π

T_start = 1
T_end = 1.001
T_delta = 0.001
# T_list = np.arange(T_start, T_end + T_delta, T_delta)
T_list = np.array([1])

for t in T_list:
    getdata_success = False
    while not getdata_success:
        time.sleep(10) # Wait 10s at the start of each query
        print(f"Querying time: {t:.3f}...")
        points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, [z], indexing='ij')], dtype=np.float64).T
        try:
            result = getData(dataset, variable, t, temporal_method, spatial_method, spatial_operator, points)
            result_array = np.array(result)
            #
            # Convert array to pandas DataFrame, and save to csv file
            #
            column_names = ['x', 'y', 'z', 'ux', 'uy', 'uz']
            dataframe = pd.DataFrame(np.hstack((points, result_array[0])), columns=column_names)
            csv_filename = dataset_title + '_t{0:.3f}'.format(t) + '.csv'
            dataframe.to_csv(csv_filename, index=False)

            u_field = np.array(result[0])[:, 0].reshape((nx, ny))
            plt.imsave(f"{save_img_dir}/slice_t{t:.3f}.png", u_field, cmap='seismic', vmin=-3, vmax=3)
            getdata_success = True
        except Exception as e:
            print(f"Warning: Failed at t={t:.3f} due to {e}. Trying again.")
