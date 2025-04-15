# === Turbulent Flow Forecasting from Saved Image Dataset ===
# Uses pre-saved training_slices (every 0.01s from 0 to 5s) to predict 0.2s ahead

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob

# === PARAMETERS ===
save_img_dir = './training_slices'
img_size = 128  # assumes 128x128 images
seq_len = 3
forecast_gap = 20  # since dt=0.01s, 0.2s = 20 steps

# === Load Data from PNGs ===
img_paths = sorted(glob.glob(os.path.join(save_img_dir, 'slice_t*.png')))

# Convert all images to tensors
slices = []
for img_path in img_paths:
    img = Image.open(img_path).convert('L')  # grayscale
    img = np.array(img).astype(np.float32) / 255.0  # scale 0-1
    img = (img - 0.5) * 6.0  # rescale to roughly [-3, 3] assuming vmin/vmax used in saving
    slices.append(img)

slices = np.array(slices)
print("Loaded shape:", slices.shape)

# === Normalize ===
mean = slices.mean()
std = slices.std()
slices = (slices - mean) / std

# === Dataset ===
class TurbulenceForecastDataset(Dataset):
    def __init__(self, data, seq_len, forecast_gap):
        self.data = data
        self.seq_len = seq_len
        self.forecast_gap = forecast_gap

    def __len__(self):
        return len(self.data) - self.seq_len - self.forecast_gap

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_len]  # (seq_len, H, W)
        target = self.data[idx + self.seq_len + self.forecast_gap - 1]  # predict t+0.2s
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

dataset = TurbulenceForecastDataset(slices, seq_len=seq_len, forecast_gap=forecast_gap)
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
        self.proj = nn.Linear(32 * img_size * img_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=2
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32 * img_size * img_size),
            nn.Unflatten(1, (32, img_size, img_size)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # (B, T, 1, H, W)
        x = x.view(B * T, 1, H, W)
        x = self.encoder(x)
        x = x.view(B, T, -1)
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        x = self.decoder(x).squeeze(1)
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
plt.title("Target Frame (t+0.2s)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap='seismic', vmin=-3, vmax=3)
plt.title("Predicted Frame")
plt.colorbar()

plt.tight_layout()
plt.show()
