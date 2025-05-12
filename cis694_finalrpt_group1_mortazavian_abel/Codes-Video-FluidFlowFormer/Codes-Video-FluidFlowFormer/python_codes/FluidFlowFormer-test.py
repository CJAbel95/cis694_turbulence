import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms.functional import normalize

# DEVICE SETUP 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Dataset Class for Turbulence Forecasting 
class TurbulenceForecastDataset(Dataset):
    def __init__(self, folder, seq_len=10, gap=100, size=256):  
        # Initialize dataset with image folder path, sequence length, frame gap, and image size
        self.paths = sorted(glob.glob(os.path.join(folder, '*.png')))  # Get sorted list of image paths
        self.seq_len = seq_len  # Number of input images in the sequence
        self.gap = gap          # Number of frames between input sequence and target frame
        self.size = size        # Resize images to (size, size)

    def __len__(self):
        # Total number of samples: exclude the last few frames that can't form a full input+target
        return len(self.paths) - self.seq_len - self.gap

    def __getitem__(self, idx):
        input_imgs = []  # List to hold input sequence of images

        # Loop over the sequence length to load input images
        for i in range(self.seq_len):
            # Open and process each image in the input sequence
            img = Image.open(self.paths[idx + i]).convert('RGB').resize((self.size, self.size))
            img = np.array(img).astype(np.float32) / 255.0             # Normalize pixel values to [0, 1]
            input_imgs.append(img.transpose(2, 0, 1))                 

        # Load the target image (future frame) after a gap
        target_img = Image.open(self.paths[idx + self.seq_len + self.gap - 1]).convert('RGB').resize((self.size, self.size))
        target_img = np.array(target_img).astype(np.float32) / 255.0   # Normalize
        target_img = target_img.transpose(2, 0, 1)                    

        # Return tensors of the input image sequence and the target image
        return torch.tensor(np.stack(input_imgs)), torch.tensor(target_img)

# Residual convolutional block
# This block is used in the encoder to extract features from the input images.
class ResidualConvBlock(nn.Module):
    """
    A residual convolutional block with normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.block(x)
        return self.relu(out + identity)

# Encoder
class FluidFlowFormerEncoder(nn.Module):
    """
    An advanced CNN-based encoder designed for turbulent fluid flow data.
    It extracts multi-scale feature representations and projects key and value maps
    for downstream attention-based modules like FlowFormer.
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()

        # Stage 1: Initial low-level feature extraction
        self.stage1 = ResidualConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)  # Output: [B, 64, H/2, W/2]

        # Stage 2: Intermediate feature representation
        self.stage2 = ResidualConvBlock(64, 128, kernel_size=5, stride=2, padding=2)          # Output: [B, 128, H/4, W/4]

        # Stage 3: Final embedding stage before attention modules
        self.stage3 = ResidualConvBlock(128, embed_dim, kernel_size=3, stride=2, padding=1)    # Output: [B, embed_dim, H/8, W/8]

        # Projection heads for attention-based modules
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # Apply hierarchical convolutional feature extraction
        skip1 = self.stage1(x)         # Low-level features
        skip2 = self.stage2(skip1)     # Mid-level features
        feat  = self.stage3(skip2)     # High-level feature map

        # Project features for key-value pairs (used in attention)
        K = self.key_proj(feat)
        V = self.value_proj(feat)

        # Return main feature, attention-ready keys and values, and skip connections
        return feat, K, V, [skip1, skip2, feat]


# Attention Block
class CostAttention(nn.Module):
    """
    Implements a spatial attention mechanism to enhance feature representations
    by computing a context vector based on similarities between a query feature map
    and provided key/value maps. Typically used in flow or feature matching tasks.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Project the input query feature to a new query embedding space
        self.query_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Scaling factor for dot-product attention to avoid large magnitude values
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values):
        """
        Arguments:
            query:   [B, C, H, W]  - The feature map that attends to others
            keys:    [B, C, H, W]  - Key features to compare against
            values:  [B, C, H, W]  - Values to weigh and combine

        Returns:
            context: [B, C, H, W]  - Weighted combination of values using attention
        """
        B, C, H, W = query.shape

        # Project and flatten the query: [B, C, H, W] -> [B, HW, C]
        query_proj = self.query_proj(query).view(B, C, -1).permute(0, 2, 1)

        # Flatten keys for dot product: [B, C, H, W] -> [B, C, HW]
        keys_flat = keys.view(B, C, -1)

        # Compute attention weights via scaled dot-product: [B, HW, HW]
        attn = torch.bmm(query_proj, keys_flat) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Flatten values and permute for weighted sum: [B, C, HW] -> [B, HW, C]
        values_flat = values.view(B, C, -1).permute(0, 2, 1)

        # Weighted combination using attention: [B, HW, HW] x [B, HW, C] -> [B, HW, C]
        context = torch.bmm(attn, values_flat)

        # Reshape back to spatial feature map: [B, HW, C] -> [B, C, H, W]
        context = context.permute(0, 2, 1).view(B, C, H, W)

        return context

# ConvGRU
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
    

# ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    """
    Implements a single ConvLSTM cell that processes spatial data with temporal recurrence.
    Input and hidden states are both 4D tensors: [batch, channels, height, width]
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        
        # Compute padding to preserve input spatial resolution
        padding = kernel_size // 2
        
        # Save hidden state dimensionality
        self.hidden_dim = hidden_dim

        # Define a single convolutional layer that outputs 4 * hidden_dim channels
        # These will be split into input, forget, output gates, and cell candidate
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, states):
        """
        Arguments:
            x      : [B, C, H, W] - Current input feature map
            states : (h_prev, c_prev) - Previous hidden and cell states

        Returns:
            h_next : Next hidden state
            c_next : Next cell state
        """
        h_prev, c_prev = states

        # Initialize hidden state if it's the first timestep
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)
        # Initialize cell state if it's the first timestep
        if c_prev is None:
            c_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)

        # Concatenate current input and previous hidden state along the channel dimension
        combined = torch.cat([x, h_prev], dim=1)  # [B, C+H, H, W]

        # Apply convolution to get all gate values at once
        conv_out = self.conv(combined)  # [B, 4*H, H, W]

        # Split the convolution output into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)

        # Input gate
        i = torch.sigmoid(cc_i)
        # Forget gate
        f = torch.sigmoid(cc_f)
        # Output gate
        o = torch.sigmoid(cc_o)
        # Cell gate (candidate values)
        g = torch.tanh(cc_g)

        # Update the cell state
        c_next = f * c_prev + i * g
        # Compute the next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
        
# Spatial Transformer Block
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

# Decoder with ConvLSTM
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


# Full Model
class FluidFlowFormer(nn.Module):
    """
    The FluidFlowFormer model processes a sequence of turbulent flow frames using a CNN-based encoder
    and a ConvLSTM-attention decoder to predict the future frame. It captures spatiotemporal dependencies
    implicitly through recurrent hidden states and attention mechanisms—without explicitly estimating
    optical flow or motion fields between frames.
    """
    def __init__(self, encoder_dim=256, hidden_dim=256, out_channels=3):
        super().__init__()
        # CNN-based encoder to extract feature maps and attention keys/values
        self.encoder = FluidFlowFormerEncoder(in_channels=3, embed_dim=encoder_dim)

        # Decoder with ConvLSTM and attention to reconstruct the next frame
        self.decoder = CostMemoryDecoderWithSkip(hidden_dim=hidden_dim, out_channels=out_channels)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h, c = None, None  # ConvLSTM hidden and cell states
        output_rgb = None  # To store final output frame

        for t in range(T):
            # Encode current frame
            _, keys, values, skips = self.encoder(x[:, t])  

            # Initialize hidden state on first timestep
            if h is None:
                h = torch.zeros_like(keys)
                c = torch.zeros_like(keys)

            # Set the current query for attention as the key (self-attention style)
            cost_query = keys

            # Decode: generate RGB output and updated hidden states
            output_rgb, (h, c), _ = self.decoder(cost_query, keys, values, None, (h, c), skips)

        return output_rgb


# Loss Functions 

# Gradient Loss
def gradient_loss(pred, target):
    # Calculate gradients along x and y directions
    pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
    target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]

    # Compute L1 loss between gradient components
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

# VGG Perceptual + L1 Loss
class PerceptualL1Loss(nn.Module):
    def __init__(self, feature_layers=['relu2_2'], weight=1.0):
        super().__init__()
        # Load pretrained VGG19 features up to relu3_1
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Convert grayscale to 3-channel if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize using ImageNet stats for VGG input
        pred_norm = normalize(pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target_norm = normalize(target, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Extract VGG features and compute perceptual loss
        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)

        loss_feat = F.l1_loss(feat_pred, feat_target)
        loss_l1 = self.l1(pred, target)

        return loss_l1 + self.weight * loss_feat

# Initialize perceptual loss globally
perceptual_loss_fn = PerceptualL1Loss(weight=1.0).to('cuda' if torch.cuda.is_available() else 'cpu')

# Full Hybrid Loss
def hybrid_loss(pred, target, alpha=0.6, beta=0.2, gamma=0.2):
    """
    This hybrid loss combines different losses with the following weights:
      - Perceptual + L1 loss (alpha)
      - SSIM loss (beta)
      - Gradient loss (gamma)
    All losses assume input range is [0, 1].
    """
    loss_perc = perceptual_loss_fn(pred, target)                  # L1 + VGG
    loss_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    loss_grad = gradient_loss(pred, target)

    return alpha * loss_perc + beta * loss_ssim + gamma * loss_grad

# CONFIG 
test_dir = './testing_slices'
model_path = 'fluidflowformer_model.h5'
img_size = 256
seq_len = 10
forecast_gap = 100  # 0.2s

# DataLoader
test_dataset = TurbulenceForecastDataset(test_dir, seq_len=seq_len, gap=forecast_gap, size=img_size)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize and Load Model
model = FluidFlowFormer().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Predict One Batch
inputs, target = next(iter(test_loader))
inputs, target = inputs.to(device), target.to(device)
target = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)

with torch.no_grad():
    prediction = model(inputs)

# Visualization
pred_img = prediction.squeeze().cpu().numpy().transpose(1, 2, 0)
target_img = target.squeeze().cpu().numpy().transpose(1, 2, 0)
input_img = inputs[0, -1].cpu().numpy().transpose(1, 2, 0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(input_img)
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
