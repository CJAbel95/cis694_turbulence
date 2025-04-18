# === ViT Forecaster with Static Spatial Embedding and Temporal Attention ===

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import timm

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# === PARAMETERS ===
train_dir = './training_slices'
test_dir = './testing_slices'
img_size = 256
seq_len = 5
forecast_gap = 5
rollout_steps = 2
batch_size = 2
epochs = 30

# === Load Slices from Folder ===
def load_slices_from_folder(folder):
    paths = sorted(glob.glob(os.path.join(folder, 'slice_t*.png')))
    images = []
    for path in paths:
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) * 6.0
        images.append(img)
    return np.array(images)

train_slices = load_slices_from_folder(train_dir)
test_slices = load_slices_from_folder(test_dir)
print("Train shape:", train_slices.shape, "Test shape:", test_slices.shape)

mean = train_slices.mean()
std = train_slices.std()
train_slices = (train_slices - mean) / std
test_slices = (test_slices - mean) / std

class TurbulenceForecastDataset(Dataset):
    def __init__(self, data, seq_len, forecast_gap):
        self.data = data
        self.seq_len = seq_len
        self.forecast_gap = forecast_gap

    def __len__(self):
        return len(self.data) - self.seq_len - forecast_gap * rollout_steps

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_len]  # [T, H, W]
        target_seq = [
            self.data[idx + self.seq_len + i * self.forecast_gap - 1] for i in range(rollout_steps)
        ]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(np.stack(target_seq), dtype=torch.float32)
        )

train_dataset = TurbulenceForecastDataset(train_slices, seq_len, forecast_gap)
test_dataset = TurbulenceForecastDataset(test_slices, seq_len, forecast_gap)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === ViT Model ===
class ViTForecaster(nn.Module):
    def __init__(self, seq_len, img_size=256):
        super().__init__()
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            in_chans=seq_len,
            img_size=img_size,
            num_classes=img_size * img_size
        )

    def forward(self, x):
        x = self.vit(x)
        return x.view(-1, 1, img_size, img_size)

model = ViTForecaster(seq_len=seq_len, img_size=img_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# === Training Loop ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)  # (B, T, H, W)
        targets = targets[:, -1].unsqueeze(1).to(device)  # last rollout target (B, 1, H, W)
        preds = model(inputs)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    input_seq, target_seq = test_dataset[0]
    input_seq = input_seq.unsqueeze(0).to(device)
    target = target_seq[-1].unsqueeze(0).unsqueeze(0).to(device)
    pred = model(input_seq)

prediction = pred.squeeze().cpu()
sample_target = target.squeeze().cpu()
sample_input = input_seq.squeeze(0).cpu()

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