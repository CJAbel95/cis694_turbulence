import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# === CONFIGURATION ===
local_filename = "isotropic1024fine.h5"

# === STEP 1: CHECK IF FILE EXISTS ===
if not os.path.exists(local_filename):
    print(f"Error: {local_filename} not found in the current directory.")
    print("Please download the file manually and place it in the same directory as this script.")
    exit(1)
else:
    print(f"{local_filename} found. Proceeding to load the data.")

# === STEP 2: LOAD DATA FROM HDF5 FILE ===
with h5py.File(local_filename, 'r') as f:
    print("Available datasets:", list(f.keys()))
    
    # Load 3D velocity field components
    u = f['Velocity_0001'][:] if 'Velocity_0001' in f else None
    v = f['Velocity_0002'][:] if 'Velocity_0002' in f else None
    w = None  # Assuming no third velocity component is present

# === STEP 3: VISUALIZE DATA (OPTIONAL) ===
if u is not None:
    plt.imshow(u[0, :, :], cmap='viridis')  # Visualize the first slice of the u-component
    plt.colorbar()
    plt.title("Velocity_0001 (first slice)")
    plt.show()

# === STEP 4: PREP FOR DEEP LEARNING ===
# Stack channels if available
channels = [u]
if v is not None: channels.append(v)
if w is not None: channels.append(w)

# Ensure channels are valid
if None in channels:
    print("Error: One or more velocity components could not be loaded.")
    exit(1)

# Shape: (C, X, Y, Z)
X = np.stack(channels, axis=0)

# Normalize
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
print("Tensor shape:", X_tensor.shape)  # (C, X, Y, Z)
