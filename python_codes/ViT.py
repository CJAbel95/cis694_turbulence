# === ViT Forecasting of Turbulent Flow 0.2s Ahead ===
# Run with:
# "C:/Users/Javad Mortazavian/anaconda3/envs/torch_gpu/python.exe" "c:/Users/Javad Mortazavian/Documents/GitHub/cis694_turbulence/vit_forecast_next0.2s.py"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import timm  # Vision Transformer models

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# === PARAMETERS ===
save_img_dir = './training_slices'
img_size = 128
seq_len = 3
forecast_gap = 50  # 0.2s ahead, since dt = 0.01s
batch_size = 2
epochs = 30

# === Load Data from PNGs ===
img_paths = sorted(glob.glob(os.path.join(save_img_dir, 'slice_t*.png')))
slices = []

for img_path in img_paths:
    img = Image.open(img_path).convert('L')  # grayscale
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) * 6.0  # scale to [-3, 3]
    slices.append(img)

slices = np.array(slices)
print("Loaded shape:", slices.shape)

# === Normalize ===
mean = slices.mean()
std = slices.std()
slices = (slices - mean) / std

# === Dataset Class ===
class TurbulenceForecastDataset(Dataset):
    def __init__(self, data, seq_len, forecast_gap):
        self.data = data
        self.seq_len = seq_len
        self.forecast_gap = forecast_gap

    def __len__(self):
        return len(self.data) - self.seq_len - self.forecast_gap

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_len]  # (seq_len, H, W)
        target = self.data[idx + self.seq_len + self.forecast_gap - 1]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

dataset = TurbulenceForecastDataset(slices, seq_len=seq_len, forecast_gap=forecast_gap)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Vision Transformer Model ===
class ViTForecaster(nn.Module):
    def __init__(self, seq_len, img_size=256, patch_size=16, emb_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.img_size = img_size
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            in_chans=seq_len,
            img_size=img_size,
            num_classes=img_size * img_size
        )

    def forward(self, x):
        # x: (B, T, H, W) → ViT expects (B, C, H, W), here C = seq_len
        x = x.to(device)
        x = self.vit(x)  # output shape: (B, H*W)
        x = x.view(-1, self.img_size, self.img_size)
        return x

# === Model, Optimizer, Loss ===
model = ViTForecaster(seq_len=seq_len, img_size=img_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# === Training Loop ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)       # (B, T, H, W)
        targets = targets.to(device)     # (B, H, W)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

# === Visual Comparison ===
model.eval()
with torch.no_grad():
    sample_input, sample_target = dataset[0]
    sample_input = sample_input.unsqueeze(0).to(device)
    prediction = model(sample_input).squeeze(0).cpu()
    sample_target = sample_target.cpu()
    sample_input = sample_input.cpu()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_input[0, -1], cmap='seismic', vmin=-3, vmax=3)
plt.title("Input (last frame)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(sample_target, cmap='seismic', vmin=-3, vmax=3)
plt.title("Target Frame (t+0.2s)")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(prediction, cmap='seismic', vmin=-3, vmax=3)
plt
