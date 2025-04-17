#
# Filename: database_extractor_gradients.py
#
# Description: Extract Velocity gradients from JHTDB Dataset.
#               Modified to include saving extracted data to csv file.
#
# Revision History
# ----------------
#   04/16/2025 -- Original. Copied from database_extractor.py
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
# auth_token = 'edu.jhu.pha.turbulence.testing-201406'
auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
# dataset_title = 'channel5200'
dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'
save_img_dir = '../Isotropic turbulence/training_slices/Iso1024CoarseVelGradient'
variable = 'velocity'
spatial_method = 'fd4lag4'
temporal_method = 'none' # 'none' or 'pchip'
spatial_operator = 'gradient'

# Create folder to save training data images
os.makedirs(save_img_dir, exist_ok=True)

# === Instantiate Dataset ===
dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)

# === Generate 2D u-velocity-gradient slices over time ===
nx = ny = 128
x_points = np.linspace(0, 8*np.pi , nx)
y_points = np.linspace(-1, 1, ny)
z = 1.5*np.pi # midplane at z = π

T_start = 0.0
T_end = 5.0
T_delta = 0.01
T_list = np.arange(T_start, T_end + T_delta, T_delta)
# T_list = np.array([1])

for t in T_list:
    getdata_success = False
    while not getdata_success:
        time.sleep(10) # Wait 10s at the start of each query
        print(f"Querying time: {t:.3f}...")
        points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, [z], indexing='ij')], dtype=np.float64).T
        try:
            result = getData(dataset, variable, t, temporal_method, spatial_method, spatial_operator, points)
            result_array = np.array(result)
            print(f"result_array.shape = {result_array.shape}")
            #
            # Convert array to pandas DataFrame, and save to csv file
            #
            column_names = ['x', 'y', 'z', 'duxdx', 'duxdy', 'duxdz', 'duydx', 'duydy', 'duydz',
                            'duzdx', 'duzdy', 'duzdz']
            dataframe = pd.DataFrame(np.hstack((points, result_array[0])), columns=column_names)
            if (t > 0.9) and (t < 1.1):
                csv_filename = dataset_title + '_t{0:.3f}'.format(t) + '.csv'
                dataframe.to_csv(csv_filename, index=False)

            duxdx = np.array(result[0])[:, 0].reshape((nx, ny))
            duxdy = np.array(result[0])[:, 1].reshape((nx, ny))
            plt.imsave(f"{save_img_dir}/slice_t{t:.3f}_duxdx.png", duxdx, cmap='seismic', vmin=-15, vmax=15)
            plt.imsave(f"{save_img_dir}/slice_t{t:.3f}_duxdy.png", duxdy, cmap='seismic', vmin=-15, vmax=15)
            getdata_success = True
        except Exception as e:
            print(f"Warning: Failed at t={t:.3f} due to {e}. Trying again.")
