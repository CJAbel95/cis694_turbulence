# === RAFT Optical Flow Forecaster for Turbulence ===

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === PARAMETERS ===
train_dir = './training_slices'
test_dir = './testing_slices'
img_size = 256
seq_len = 5
forecast_gap = 20
rollout_steps = 2
batch_size = 2
epochs = 15

# === LOAD DATA ===
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
mean, std = train_slices.mean(), train_slices.std()
train_slices = (train_slices - mean) / std
test_slices = (test_slices - mean) / std

# === DATASET ===
class TurbulenceForecastDataset(Dataset):
    def __init__(self, data, seq_len, forecast_gap):
        self.data = data
        self.seq_len = seq_len
        self.forecast_gap = forecast_gap

    def __len__(self):
        return len(self.data) - self.seq_len - forecast_gap * rollout_steps

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        target_seq = [
            self.data[idx + self.seq_len + i * self.forecast_gap - 1]
            for i in range(rollout_steps)
        ]
        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(np.stack(target_seq), dtype=torch.float32)
        )

train_dataset = TurbulenceForecastDataset(train_slices, seq_len, forecast_gap)
test_dataset = TurbulenceForecastDataset(test_slices, seq_len, forecast_gap)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === RAFT COMPONENTS ===
class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class CorrelationLayer(nn.Module):
    def forward(self, fmap1, fmap2):
        B, C, H, W = fmap1.shape
        fmap1 = fmap1.view(B, C, H * W)
        fmap2 = fmap2.view(B, C, H * W)
        corr = torch.bmm(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(B, H, W, H, W)
        corr = corr.mean(dim=(-1, -2))  # simplified for speed
        return corr.unsqueeze(1)  # B x 1 x H x W

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.convz = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(combined))
        r = torch.sigmoid(self.convr(combined))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h

class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = ConvGRU(input_dim=1 + 2, hidden_dim=hidden_dim)  # ✅ FIXED INPUT DIM
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(self, net, corr, flow):
        x = torch.cat([corr, flow], dim=1)
        net = self.gru(x, net)
        delta_flow = self.flow_head(net)
        return net, delta_flow

class RAFT(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fnet = FeatureEncoder()
        self.corr = CorrelationLayer()
        self.update_block = UpdateBlock(hidden_dim)

    def initialize_flow(self, img):
        N, C, H, W = img.size()
        coords = torch.meshgrid(torch.arange(H, device=img.device),
                                torch.arange(W, device=img.device), indexing='ij')
        coords = torch.stack(coords[::-1], dim=0).float()
        coords = coords.unsqueeze(0).repeat(N, 1, 1, 1)
        return coords, torch.zeros_like(coords)

    def forward(self, image1, image2, iters=12):
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        coords0, flow = self.initialize_flow(image1)
        flow = F.interpolate(flow, size=fmap1.shape[-2:], mode='bilinear', align_corners=False)
        corr = self.corr(fmap1, fmap2)
        net = torch.tanh(torch.zeros_like(fmap1)).to(image1.device)

        for _ in range(iters):
            net, delta_flow = self.update_block(net, corr, flow)
            flow = flow + delta_flow

        return F.interpolate(flow, size=image1.shape[-2:], mode='bilinear', align_corners=False)

def warp(img, flow):
    B, C, H, W = img.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=img.device),
                                    torch.arange(W, device=img.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    flow = flow.permute(0, 2, 3, 1)
    warped_grid = grid + flow
    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (W - 1) - 1.0
    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (H - 1) - 1.0
    return F.grid_sample(img, warped_grid, align_corners=True)

# === TRAINING ===
raft = RAFT().to(device)
optimizer = optim.Adam(raft.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(epochs):
    raft.train()
    total_loss = 0
    for inputs, targets in train_loader:
        last_input = inputs[:, -1:].to(device)
        target = targets[:, -1:].to(device)
        flow = raft(last_input, target)
        pred = warp(last_input, flow)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

# === EVALUATION & VISUALIZATION ===
raft.eval()
with torch.no_grad():
    input_seq, target_seq = test_dataset[0]
    input_seq = input_seq.unsqueeze(0).to(device)
    target = target_seq[-1].unsqueeze(0).unsqueeze(0).to(device)
    last_input = input_seq[:, -1:]
    flow = raft(last_input, target)
    pred = warp(last_input, flow)

    prediction = pred.squeeze().cpu()
    sample_target = target.squeeze().cpu()
    sample_input = input_seq.squeeze(0)[-1].cpu()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_input, cmap='seismic', vmin=-3, vmax=3)
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
