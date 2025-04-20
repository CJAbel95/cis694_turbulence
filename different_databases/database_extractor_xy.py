import numpy as np
import matplotlib.pyplot as plt
import os
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData

# === PARAMETERS ===
auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
dataset_title = 'isotropic1024coarse' # 'transition_to_turbulence', 'channel5200', 'isotropic1024coarse', 'isotropic1024fine'
#dataset_title = 'channel5200'
#dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'
save_img_dir = './different_databases/training_slices_isotropic1024coarse'
variable = 'velocity'
spatial_method = 'none'
temporal_method = 'pchip' # 'none' or 'pchip'
spatial_operator = 'field'

# Create folder to save training data images
os.makedirs(save_img_dir, exist_ok=True)

# === Instantiate Dataset ===
dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)

# === Generate 2D u-velocity slices over time ===
nx = ny = 256
x_points = np.linspace(0, 2*np.pi , nx)
y_points = np.linspace(0, 2*np.pi, ny)
z = np.pi # midplane at z = π

T_start = 0
T_end = 10
T_delta = 0.002
T_list = np.arange(T_start, T_end + T_delta, T_delta)

for t in T_list:
    print(f"Querying time: {t:.3f}...")
    points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, [z], indexing='ij')], dtype=np.float64).T
    try:
        result = getData(dataset, variable, t, temporal_method, spatial_method, spatial_operator, points)
        u_field = np.array(result[0])[:, 0].reshape((nx, ny))
        plt.imsave(f"{save_img_dir}/slice_t{t:.3f}.png", u_field, cmap='seismic', vmin=-3, vmax=3)
    except Exception as e:
        print(f"Warning: Failed at t={t:.3f} due to {e}. Skipping.")
