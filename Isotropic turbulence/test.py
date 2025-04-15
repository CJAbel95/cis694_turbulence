# === Turbulent Flow Forecasting with Correct Domain Units (0 to 2π) ===
# Updated to extract physically meaningful velocity slices for DL training

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData
import os
import time

# === PARAMETERS ===
auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
dataset_title = 'channel5200'
output_path = './giverny_output'
save_img_dir = './training_slices'
variable = 'velocity'
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

T_start = 0.0
T_end = 5
T_delta = 0.01
T_list = np.arange(T_start, T_end + T_delta, T_delta)

slices = []

for i, t in enumerate(T_list):
    print(f"Querying time: {t:.3f}...")
    points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, [z], indexing='ij')], dtype=np.float64).T
    try:
        result = getData(dataset, variable, t, temporal_method, spatial_method, spatial_operator, points)
        u_field = np.array(result[0])[:, 0].reshape((nx, ny))
        slices.append(u_field)

        # Save to image
        plt.imsave(f"{save_img_dir}/slice_t{t:.3f}.png", u_field, cmap='seismic', vmin=-3, vmax=3)
    except Exception as e:
        print(f"Warning: Failed at t={t:.3f} due to {e}. Skipping.")
        continue

slices = np.array(slices)
print("Shape of dataset:", slices.shape)

# === Normalize ===
mean = slices.mean()
std = slices.std()
slices = (slices - mean) / std

# === Dataset Preparation ===
class TurbulenceDataset(Dataset):
    def __init__(self, data, seq_len=3):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]  # (seq_len, H, W)
        target = self.data[idx + self.seq_len]          # (H, W)
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

seq_len = 3
dataset = TurbulenceDataset(slices, seq_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# === CNN + Transformer Architecture ===
class ConvTransformer(nn.Module):
    def __init__(self, seq_len, in_channels=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(32 * nx * ny, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32 * nx * ny),
            nn.Unflatten(1, (32, nx, ny)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # (B, T, 1, H, W)
        x = x.view(B * T, 1, H, W)  # reshape for conv
        x = self.encoder(x)  # (B*T, C, H, W)
        x = x.view(B, T, -1)  # (B, T, flat)
        x = self.proj(x)      # (B, T, hidden_dim)
        x = x.permute(1, 0, 2)  # (T, B, hidden)
        x = self.transformer(x)  # (T, B, hidden)
        x = x[-1]  # last timestep
        x = self.decoder(x).squeeze(1)  # (B, H, W)
        return x

# === Train Model ===
model = ConvTransformer(seq_len)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

model.train()
for epoch in range(30):
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch  # (B, T, H, W), (B, H, W)
        preds = model(inputs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# === Visual Comparison ===
model.eval()
with torch.no_grad():
    sample_input, sample_target = dataset[0]
    prediction = model(sample_input.unsqueeze(0)).squeeze(0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_input[-1], cmap='seismic', vmin=-3, vmax=3)
plt.title("Input (last frame)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sample_target, cmap='seismic', vmin=-3, vmax=3)
plt.title("True Next Frame")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap='seismic', vmin=-3, vmax=3)
plt.title("Predicted Frame")
plt.colorbar()

plt.tight_layout()
plt.show()
