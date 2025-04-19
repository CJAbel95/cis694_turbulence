#
# SuperResNetworks.py
# Description -- Transform and Dataset for Superresolution
#                application with JHTDB data.
#
# by Christopher Abel
# Revision History
# ----------------
#   04/16/2025 -- Original
#
# -------------------------------------------------
import os
import numpy as np
import torch
from torchvision.transforms import ToTensor, v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def conv_relu2x_blk(in_chan, out_chan):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_chan, out_chan, 3, 1, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_chan, out_chan, 3, 1, 1),
        torch.nn.ReLU()
    )


class BaseSRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Stages in contracting path
        self.convd1 = conv_relu2x_blk(4, 4)
        self.convd2 = conv_relu2x_blk(4, 8)
        self.convd3 = conv_relu2x_blk(8, 16)
        self.convd4 = conv_relu2x_blk(16, 32)
        self.convCtr = conv_relu2x_blk(32, 64)
        self.maxpool = torch.nn.MaxPool2d(2, 2, 0)
        # Stages in expanding path
        self.upscale1 = torch.nn.ConvTranspose2d(64, 32, 2, 2)
        self.upscale2 = torch.nn.ConvTranspose2d(32, 16, 2, 2)
        self.upscale3 = torch.nn.ConvTranspose2d(16, 8, 2, 2)
        self.upscale4 = torch.nn.ConvTranspose2d(8, 4, 2, 2)
        self.upscale5 = torch.nn.ConvTranspose2d(4, 4, 2, 2)
        self.upscale6 = torch.nn.ConvTranspose2d(4, 4, 2, 2)
        self.convu1 = conv_relu2x_blk(2 * 32, 32)
        self.convu2 = conv_relu2x_blk(2 * 16, 16)
        self.convu3 = conv_relu2x_blk(2 * 8, 8)
        self.convu4 = conv_relu2x_blk(2 * 4, 4)
        self.convu5 = conv_relu2x_blk(4, 4)
        self.convu6 = conv_relu2x_blk(4, 4)
        self.final = torch.nn.Conv2d(4, 4, 3, 1, 1)


    def forward(self, x):
        h1 = self.convd1(x)
        h1p = self.maxpool(h1)
        h2 = self.convd2(h1p)
        h2p = self.maxpool(h2)
        h3 = self.convd3(h2p)
        h3p = self.maxpool(h3)
        h4 = self.convd4(h3p)
        h4p = self.maxpool(h4)
        h5 = self.convCtr(h4p)
        h5u = self.upscale1(h5)
        h5c = torch.cat([h5u, h4], dim=1)
        h6 = self.convu1(h5c)
        h6u = self.upscale2(h6)
        h6c = torch.cat([h6u, h3], dim=1)
        h7 = self.convu2(h6c)
        h7u = self.upscale3(h7)
        h7c = torch.cat([h7u, h2], dim=1)
        h8 = self.convu3(h7c)
        h8u = self.upscale4(h8)
        h8c = torch.cat([h8u, h1], dim=1)
        h9 = self.convu4(h8c)
        h9u = self.upscale5(h9)
        h10 = self.convu5(h9u)
        h10u = self.upscale6(h10)
        h11 = self.convu6(h10u)
        h12 = self.final(h11)
        return h12


class SuperResTransform:
    def __init__(self, size):
        self.transform = v2.Compose([
            v2.Resize((size, size))
        ])


class SuperResDataset(Dataset):
    def __init__(self, img_dir, transform_orig=None,
                 transform1=None, target_transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(self.img_dir) if
                          os.path.isfile(os.path.join(self.img_dir, f)) and f.endswith('.png')]
        self.transform_orig = transform_orig
        self.transform1 = transform1
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path)
        # Transform image to tensor, and save original version to return
        img = v2.functional.to_image(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        if self.transform_orig:
            img_orig = self.transform_orig(img)
        else:
            img_orig = img.clone()
        # Apply transform to image, if defined
        if self.transform1:
            img1 = self.transform1(img)
        else:
            img1 = img
        return img_orig, img1


def initialize_weights(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, mean=0.025, std=0.01)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(module.weight, mean=0.025, std=0.01)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)


def main():
    imagepath = '../Isotropic turbulence/training_slices/'
    #
    # Test image scaling transform and SuperResDataset
    #
    transform1 = SuperResTransform(32).transform
    dataset = SuperResDataset(imagepath, transform1=transform1)
    print(f"There are {len(dataset)} images in {imagepath}")
    image_orig, image_scaled = dataset[0]
    fig1, (ax1a, ax1b) = plt.subplots(1, 2)
    im1 = ax1a.imshow(image_orig.permute(1, 2, 0).squeeze())
    ax1a.set_title('Original')
    ax1b.imshow(image_scaled.permute(1, 2, 0).squeeze())
    ax1b.set_title('Downsampled by 4x')

    #
    # Test use of dataloader
    #
    dataloader1 = DataLoader(dataset, batch_size=4, shuffle=False)
    images_orig, images_scaled = next(iter(dataloader1))
    print(f"\nMax / Mean / Min of elements in images_orig = {torch.max(images_orig)} / "
          f"{torch.mean(images_orig)} / {torch.min(images_orig)}")
    print(f"images_orig.shape = {images_orig.shape}, images_scaled.shape = {images_scaled.shape}")
    print(f"images_orig[0].shape = {images_orig[0].shape}, images_scaled[0].shape = {images_scaled[0].shape}")
    print(f"images_orig[0, 0, 0:5, 0:5] = {images_orig[0, 0, 100:105, 100:105]}")
    print(f"images_orig[0, 1, 0:5, 0:5] = {images_orig[0, 1, 100:105, 100:105]}")
    print(f"images_orig[0, 2, 0:5, 0:5] = {images_orig[0, 2, 100:105, 100:105]}")
    print(f"images_orig[0, 3, 0:5, 0:5] = {images_orig[0, 3, 100:105, 100:105]}")
    fig2 = plt.figure()
    for i in range(0, 4):
        image_orig = np.clip(images_orig[i].permute(1, 2, 0), 0, 1)
        fig2.add_subplot(1, 4, i + 1)
        plt.imshow(image_orig.squeeze())

    #
    # Test complete network
    #
    down_net = BaseSRNet()
    down_net.eval()
    images_decode = down_net(images_scaled)
    print(f"images_decode shape: {images_decode.shape}")

    # Display all plots
    plt.show()


if __name__ == '__main__':
    main()
