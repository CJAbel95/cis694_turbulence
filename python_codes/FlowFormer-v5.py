#v2: Temporal-Aware Loss (Gradient Flow Loss) added
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms.functional import normalize

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# === Dataset ===
class TurbulenceForecastDataset(Dataset):
    def __init__(self, folder, seq_len=5, gap=20, size=256):
        self.paths = sorted(glob.glob(os.path.join(folder, '*.png')))
        self.seq_len = seq_len
        self.gap = gap
        self.size = size

    def __len__(self):
        return len(self.paths) - self.seq_len - self.gap

    def __getitem__(self, idx):
        input_imgs = []
        for i in range(self.seq_len):
            img = Image.open(self.paths[idx + i]).convert('RGB').resize((self.size, self.size))
            img = np.array(img).astype(np.float32) / 255.0
            input_imgs.append(img.transpose(2, 0, 1))
        target_img = Image.open(self.paths[idx + self.seq_len + self.gap - 1]).convert('RGB').resize((self.size, self.size))
        target_img = np.array(target_img).astype(np.float32) / 255.0
        target_img = target_img.transpose(2, 0, 1)
        return torch.tensor(np.stack(input_imgs)), torch.tensor(target_img)

# === Encoder ===
class MiniFlowFormerEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        skip1 = self.conv1(x)   # [B, 64, H/2, W/2]
        skip2 = self.conv2(skip1)  # [B, 128, H/4, W/4]
        feat = self.conv3(skip2)   # [B, 256, H/8, W/8]
        K = self.key_proj(feat)
        V = self.value_proj(feat)
        return feat, K, V, [skip1, skip2, feat]


# === Attention Block ===
class CostAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values):
        B, C, H, W = query.shape
        query_flat = self.query_proj(query).view(B, C, -1).permute(0, 2, 1)
        keys_flat = keys.view(B, C, -1)
        attn = torch.bmm(query_flat, keys_flat) * self.scale
        attn = F.softmax(attn, dim=-1)
        values_flat = values.view(B, C, -1).permute(0, 2, 1)
        context = torch.bmm(attn, values_flat).permute(0, 2, 1).view(B, C, H, W)
        return context

# === ConvGRU ===
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.convz = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.convh = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        xh = torch.cat([x, h_prev], dim=1)
        z = torch.sigmoid(self.convz(xh))
        r = torch.sigmoid(self.convr(xh))
        xrh = torch.cat([x, r * h_prev], dim=1)
        h_hat = torch.tanh(self.convh(xrh))
        h_next = (1 - z) * h_prev + z * h_hat
        return h_next
    
# === ConvLSTM ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, states):
        h_prev, c_prev = states

        # Handle first-time initialization
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
        if c_prev is None:
            c_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)

        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
        
# === Spatial Transformer Block ===
class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

# === Decoder with ConvLSTM ===
class CostMemoryDecoderWithSkip(nn.Module):
    def __init__(self, hidden_dim=256, out_channels=3):
        super().__init__()

        self.attn = CostAttention(hidden_dim)
        self.convlstm = ConvLSTMCell(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        self.transformer = SpatialTransformerBlock(dim=hidden_dim)
        self.flow_head = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim + 128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, cost_query, keys, values, flow_init, prev_states, skips):
        context = self.attn(cost_query, keys, values)
        x = torch.cat([cost_query, context], dim=1)

        if not isinstance(prev_states, (tuple, list)) or len(prev_states) != 2:
            h_prev = torch.zeros_like(cost_query)
            c_prev = torch.zeros_like(cost_query)
        else:
            h_prev, c_prev = prev_states

        h, c = self.convlstm(x, (h_prev, c_prev))
        h = self.transformer(h)
        delta_flow = self.flow_head(h)
        flow = delta_flow if flow_init is None else flow_init + delta_flow

        up1 = self.up1(torch.cat([h, skips[2]], dim=1))
        up2 = self.up2(torch.cat([up1, skips[1]], dim=1))
        up3 = self.up3(torch.cat([up2, skips[0]], dim=1))

        return up3, (h, c), flow


# === Full Model ===
class FlowFormerForecastModel(nn.Module):
    def __init__(self, encoder_dim=256, hidden_dim=256, out_channels=3):
        super().__init__()
        self.encoder = MiniFlowFormerEncoder(in_channels=3, embed_dim=encoder_dim)
        self.decoder = CostMemoryDecoderWithSkip(hidden_dim=hidden_dim, out_channels=out_channels)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h, c = None, None

        for t in range(T):
            _, keys, values, skips = self.encoder(x[:, t])  

            if h is None:
                h = torch.zeros_like(keys)
                lowres_flow = torch.zeros((B, 2, keys.shape[2], keys.shape[3]), device=keys.device)
                flow = None  # explicitly define flow on first timestep

            cost_query = keys

            if lowres_flow.shape[-2:] != keys.shape[-2:]:
                print(f"WARNING: lowres_flow shape {lowres_flow.shape} vs delta_flow shape {keys.shape}")
                lowres_flow = torch.zeros((B, 2, keys.shape[2], keys.shape[3]), device=keys.device)

            output_rgb, (h, c), flow = self.decoder(cost_query, keys, values, lowres_flow, (h, c), skips)
            lowres_flow = self.decoder.flow_head(h)  # ✅ Now h is the hidden state only


        return output_rgb


# === Loss Functions ===
def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
    target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]

    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

class PerceptualL1Loss(nn.Module):
    def __init__(self, feature_layers=['relu2_2'], weight=1.0):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()  # Up to relu3_1
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Normalize to VGG input range [0, 1] -> ImageNet stats
        pred_norm = normalize(pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_norm = normalize(target, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)

        loss_feat = F.l1_loss(feat_pred, feat_target)
        loss_l1 = self.l1(pred, target)

        return loss_l1 + self.weight * loss_feat

# === Configs ===
train_dir = './training_slices'
test_dir = './testing_slices'
img_size = 256
seq_len = 10
forecast_gap = 100 # 0.2s
batch_size = 6
epochs = 100
lr = 5e-4

# === Data ===
train_dataset = TurbulenceForecastDataset(train_dir, seq_len=seq_len, gap=forecast_gap, size=img_size)
test_dataset = TurbulenceForecastDataset(test_dir, seq_len=seq_len, gap=forecast_gap, size=img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# === Training ===
model = FlowFormerForecastModel().to(device)
# === Replace this:
# criterion = nn.MSELoss()

# === With this:
criterion = PerceptualL1Loss(weight=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, target = batch
        inputs, target = inputs.to(device), target.to(device)

        # Ensure target has shape [B, 3, 256, 256]
        target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        pred = model(inputs)

        # Compute perceptual + L1 + gradient loss
        loss_perceptual = criterion(pred, target)
        loss_grad = gradient_loss(pred, target)
        loss = loss_perceptual + 0.15 * loss_grad  # Adjust weight if needed

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(train_loader):.6f}")


# === Test & Visualize ===
model.eval()
inputs, target = next(iter(test_loader))
inputs, target = inputs.to(device), target.to(device)

# Resize target to match prediction resolution
target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)

with torch.no_grad():
    prediction = model(inputs)


pred_img = prediction.squeeze().cpu().numpy().transpose(1, 2, 0)
target_img = target.squeeze().cpu().numpy().transpose(1, 2, 0)
input_seq_img = inputs[0, -1].cpu().numpy().transpose(1, 2, 0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(input_seq_img)
plt.title("Last Input Frame")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(target_img)
plt.title("Ground Truth Future")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pred_img)
plt.title("Predicted Future")
plt.axis('off')

plt.tight_layout()
plt.show()
