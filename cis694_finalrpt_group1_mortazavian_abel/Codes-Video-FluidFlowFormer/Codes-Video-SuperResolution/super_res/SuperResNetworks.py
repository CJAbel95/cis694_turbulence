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
from torchvision.transforms import ToTensor, v2, InterpolationMode, functional
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


class ResizeTransform:
    # Transform applied to get reduced size image from original.
    def __init__(self, size):
        self.transform = v2.Compose([
            v2.Resize((size, size), interpolation=InterpolationMode.NEAREST)
        ])


class ResizeFiltTransform:
    # Transform applied to get reduced size image from original.
    def __init__(self, size):
        self.transform = v2.Compose([
            v2.Resize((size, size))
        ])


class AugmentTransform:
    # Transform applied to randomly flip or rotate image
    def __init__(self):
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.Lambda(lambda img: functional.rotate(img, 0)),
                v2.Lambda(lambda img: functional.rotate(img, 90)),
                v2.Lambda(lambda img: functional.rotate(img, 180)),
                v2.Lambda(lambda img: functional.rotate(img, 270))
            ])
        ])


class SRCNNmod(torch.nn.Module):
    #
    # Modified SRCNN model; upsample with bicubic interpolation. No Batch normalization.
    def __init__(self, n0, f1, n1, f2, n2, f3):
        super(SRCNNmod, self).__init__()
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1 = torch.nn.Conv2d(n0, n1, f1, 1, (f1 - 1) // 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x):
        xu = self.upscale(x)
        h1 = self.conv1(xu)
        h1r = self.relu1(h1)
        h2 = self.conv2(h1r)
        h2r = self.relu2(h2)
        h3 = self.conv3(h2r)
        return h3


class SRCNNmodBn(torch.nn.Module):
    #
    # Modified SRCNN model; upsample with bicubic interpolation. Includes batch
    # normalization stages.
    def __init__(self, n0, f1, n1, f2, n2, f3):
        super(SRCNNmodBn, self).__init__()
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1 = torch.nn.Conv2d(n0, n1, f1, 1, (f1 - 1) // 2)
        self.bn1 = torch.nn.BatchNorm2d(n1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.bn2 = torch.nn.BatchNorm2d(n2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x):
        xu = self.upscale(x)
        h1 = self.conv1(xu)
        h1b = self.bn1(h1)
        h1r = self.relu1(h1b)
        h2 = self.conv2(h1r)
        h2b = self.bn2(h2)
        h2r = self.relu2(h2b)
        h3 = self.conv3(h2r)
        return h3


class SRCNNmodRl(torch.nn.Module):
    #
    # Modified SRCNN model; upsample with bicubic interpolation. No Batch normalization.
    def __init__(self, n0, f1, n1, f2, n2, f3):
        super(SRCNNmodRl, self).__init__()
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1 = torch.nn.Conv2d(n0, n1, f1, 1, (f1 - 1) // 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x):
        xu = self.upscale(x)
        h1 = self.conv1(xu)
        h1r = self.relu1(h1)
        h2 = self.conv2(h1r)
        h2r = self.relu2(h2)
        h3 = self.conv3(h2r)
        return h3 + xu


class SRCNNmod2(torch.nn.Module):
    #
    # Modified SRCNN model; upsample using ConvTranspose2d
    def __init__(self, n0, f0, f1, n1, f2, n2, f3):
        super(SRCNNmod2, self).__init__()
        self.upscale = torch.nn.ConvTranspose2d(n0, n0, 4, 4, 0, 0)
        self.conv0 = torch.nn.Conv2d(n0, n0, f0, 1, (f0 - 1) // 2)
        self.relu0 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(n0, n1, f1, 1, (f1 - 1) // 2)
        self.bn1 = torch.nn.BatchNorm2d(n1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.bn2 = torch.nn.BatchNorm2d(n2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x):
        xu = self.upscale(x)
        h0 = self.conv0(xu)
        h0r = self.relu0(h0)
        h1 = self.conv1(h0r)
        h1b = self.bn1(h1)
        h1r = self.relu1(h1b)
        h2 = self.conv2(h1r)
        h2b = self.bn2(h2)
        h2r = self.relu2(h2b)
        h3 = self.conv3(h2r)
        return h3


class TwoBrSR(torch.nn.Module):
    #
    #   SRCNN model with two input branches.  Branch 1 is the image to be super-resolved,
    #   (e.g. ux) while branch 2 is a related image (e.g. p), captured at the same time
    #   point.
    #
    #   Upsample with bicubic interpolation. No Batch normalization.
    def __init__(self, n0, f1, n1, f2, n2, f3):
        super(TwoBrSR, self).__init__()
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1 = torch.nn.Conv2d(2 * n0, n1, f1, 1, (f1 - 1) // 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x, y):
        xu = self.upscale(x)
        yu = self.upscale(y)
        xy_cat = torch.cat((xu, yu), dim=1)
        h1 = self.conv1(xy_cat)
        h1r = self.relu1(h1)
        h2 = self.conv2(h1r)
        h2r = self.relu2(h2)
        h3 = self.conv3(h2r)
        return h3


class ThreeBrSR(torch.nn.Module):
    #
    #   SRCNN model with three input branches.  Branch 1 is the image to be super-resolved,
    #   (e.g. ux) while branches 2 and 3 are related images (e.g. dp/dx and dp/dy),
    #   captured at the same time point.
    #
    #   Upsample with bicubic interpolation. No Batch normalization.
    def __init__(self, n0, f1, n1, f2, n2, f3):
        super(ThreeBrSR, self).__init__()
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.conv1 = torch.nn.Conv2d(3 * n0, n1, f1, 1, (f1 - 1) // 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n1, n2, f2, 1, (f2 - 1) // 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(n2, n0, f3, 1, (f3 - 1) // 2)

    def forward(self, x, y1, y2):
        xu = self.upscale(x)
        y1u = self.upscale(y1)
        y2u = self.upscale(y2)
        xy_cat = torch.cat((xu, y1u, y2u), dim=1)
        h1 = self.conv1(xy_cat)
        h1r = self.relu1(h1)
        h2 = self.conv2(h1r)
        h2r = self.relu2(h2)
        h3 = self.conv3(h2r)
        return h3


class SuperResDataset(Dataset):
    def __init__(self, img_dir, transform_orig=None,
                 transform1=None, target_transform=None, remove_alpha=True):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(self.img_dir) if
                          os.path.isfile(os.path.join(self.img_dir, f)) and f.endswith('.png')]
        self.transform_orig = transform_orig
        self.transform1 = transform1
        self.target_transform = target_transform
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path)
        if self.remove_alpha and img.mode == 'RGBA':
            img = img.convert('RGB')
        # Transform image to tensor, and save original version to return
        img = v2.functional.to_image(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        if self.transform_orig:
            img_orig = self.transform_orig(img)
        else:
            img_orig = img.clone()
        # Apply transform to image, if defined
        if self.transform1:
            img1 = self.transform1(img_orig)
        else:
            img1 = img_orig
        return img_orig, img1


class MBSRDataset(Dataset):
    #
    #   Pytorch Dataset with two input branches
    #       branch1: primary input; this is the target image for super-resolution
    #       branch2: secondary input; a related image whose features may enhance
    #                   the SR of the primary branch.
    #
    def __init__(self, img_dir_br1, img_dir_br2, transform_orig=None,
                 transform1=None, target_transform=None, remove_alpha=True):
        self.img_dir_br1 = img_dir_br1
        self.img_dir_br2 = img_dir_br2
        self.img_files_br1 = [f for f in os.listdir(self.img_dir_br1) if
                          os.path.isfile(os.path.join(self.img_dir_br1, f)) and f.endswith('.png')]
        self.img_files_br2 = [f for f in os.listdir(self.img_dir_br2) if
                              os.path.isfile(os.path.join(self.img_dir_br2, f)) and f.endswith('.png')]
        self.transform_orig = transform_orig
        self.transform1 = transform1
        self.target_transform = target_transform
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.img_files_br1)

    def __getitem__(self, idx):
        img_path_br1 = os.path.join(self.img_dir_br1, self.img_files_br1[idx])
        img_path_br2 = os.path.join(self.img_dir_br2, self.img_files_br2[idx])
        # print(f"{img_path_br1}, {img_path_br2}")
        img_br1 = Image.open(img_path_br1)
        img_br2 = Image.open(img_path_br2)
        if self.remove_alpha and img_br1.mode == 'RGBA':
            img_br1 = img_br1.convert('RGB')
        if self.remove_alpha and img_br2.mode == 'RGBA':
            img_br2 = img_br2.convert('RGB')
        # Transform image to tensor, and save original version to return
        img_br1 = v2.functional.to_image(img_br1)
        img_br2 = v2.functional.to_image(img_br2)
        img_br1 = v2.functional.to_dtype(img_br1, torch.float32, scale=True)
        img_br2 = v2.functional.to_dtype(img_br2, torch.float32, scale=True)
        if self.transform_orig:
            img_orig_br1 = self.transform_orig(img_br1)
            img_orig_br2 = self.transform_orig(img_br2)
        else:
            img_orig_br1 = img_br1.clone()
            img_orig_br2 = img_br2.clone()
        # Apply transform to image, if defined
        if self.transform1:
            img1_br1 = self.transform1(img_orig_br1)
            img1_br2 = self.transform1(img_orig_br2)
        else:
            img1_br1 = img_orig_br1
            img1_br2 = img_orig_br2
        return img_orig_br1, img1_br1, img_orig_br2, img1_br2


class ThreeBrDataset(Dataset):
    #
    #   Pytorch Dataset with three input branches
    #       branch1: primary input; this is the target image for super-resolution
    #       branch2: secondary input; a related image whose features may enhance
    #                   the SR of the primary branch.
    #       branch3: secondary input; another related image
    #
    def __init__(self, img_dir_br1, img_dir_br2, img_dir_br3, transform_orig=None,
                 transform1=None, target_transform=None, remove_alpha=True):
        self.img_dir_br1 = img_dir_br1
        self.img_dir_br2 = img_dir_br2
        self.img_dir_br3 = img_dir_br3
        self.img_files_br1 = [f for f in os.listdir(self.img_dir_br1) if
                          os.path.isfile(os.path.join(self.img_dir_br1, f)) and f.endswith('.png')]
        self.img_files_br2 = [f for f in os.listdir(self.img_dir_br2) if
                              os.path.isfile(os.path.join(self.img_dir_br2, f)) and f.endswith('.png')]
        self.img_files_br3 = [f for f in os.listdir(self.img_dir_br3) if
                              os.path.isfile(os.path.join(self.img_dir_br3, f)) and f.endswith('.png')]
        self.transform_orig = transform_orig
        self.transform1 = transform1
        self.target_transform = target_transform
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.img_files_br1)

    def __getitem__(self, idx):
        img_path_br1 = os.path.join(self.img_dir_br1, self.img_files_br1[idx])
        img_path_br2 = os.path.join(self.img_dir_br2, self.img_files_br2[idx])
        img_path_br3 = os.path.join(self.img_dir_br3, self.img_files_br2[idx])
        # print(f"{img_path_br1}, {img_path_br2}, {img_path_br3}")
        img_br1 = Image.open(img_path_br1)
        img_br2 = Image.open(img_path_br2)
        img_br3 = Image.open(img_path_br3)
        if self.remove_alpha and img_br1.mode == 'RGBA':
            img_br1 = img_br1.convert('RGB')
        if self.remove_alpha and img_br2.mode == 'RGBA':
            img_br2 = img_br2.convert('RGB')
        if self.remove_alpha and img_br3.mode == 'RGBA':
            img_br3 = img_br3.convert('RGB')
        # Transform image to tensor, and save original version to return
        img_br1 = v2.functional.to_image(img_br1)
        img_br2 = v2.functional.to_image(img_br2)
        img_br3 = v2.functional.to_image(img_br3)
        img_br1 = v2.functional.to_dtype(img_br1, torch.float32, scale=True)
        img_br2 = v2.functional.to_dtype(img_br2, torch.float32, scale=True)
        img_br3 = v2.functional.to_dtype(img_br3, torch.float32, scale=True)
        if self.transform_orig:
            img_orig_br1 = self.transform_orig(img_br1)
            img_orig_br2 = self.transform_orig(img_br2)
            img_orig_br3 = self.transform_orig(img_br3)
        else:
            img_orig_br1 = img_br1.clone()
            img_orig_br2 = img_br2.clone()
            img_orig_br3 = img_br3.clone()
        # Apply transform to image, if defined
        if self.transform1:
            img1_br1 = self.transform1(img_orig_br1)
            img1_br2 = self.transform1(img_orig_br2)
            img1_br3 = self.transform1(img_orig_br3)
        else:
            img1_br1 = img_orig_br1
            img1_br2 = img_orig_br2
            img1_br3 = img_orig_br3
        return img_orig_br1, img1_br1, img1_br2, img1_br3


def initialize_weights(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(module.weight)  # Normal distribution
        # torch.nn.init.kaiming_normal_(module.weight)  # Normal distribution
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.01)


def main():
    imagepath = '../Isotropic turbulence/training_slices/'
    #
    # Test image augmentation transform and scaling transform
    #
    transform0 = AugmentTransform().transform
    transform1 = ResizeTransform(32).transform
    dataset = SuperResDataset(imagepath, transform_orig=transform0, transform1=transform1, remove_alpha=False)
    print(f"There are {len(dataset)} images in {imagepath}")
    image_orig, image_scaled = dataset[0]
    image_upscaled = torch.nn.functional.interpolate(image_scaled.unsqueeze(0), scale_factor=4,
                                                     mode='bicubic', align_corners=False)
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3)
    im1 = ax1a.imshow(image_orig.permute(1, 2, 0).squeeze())
    ax1a.set_title('Original')
    ax1b.imshow(image_scaled.permute(1, 2, 0).squeeze())
    ax1b.set_title('Downsampled by 4x')
    ax1c.imshow(image_upscaled.squeeze().permute(1, 2, 0))
    ax1c.set_title('Upscaled by 4x from Downscaled Image')

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
    # Test BaseSRNet
    #
    # down_net = BaseSRNet()
    # down_net.eval()
    # images_decode = down_net(images_scaled)
    # print(f"images_decode shape: {images_decode.shape}")

    #
    # Test SRCNNmod
    #
    # srcnn_net = SRCNNmod(n0=4, f1=9, n1=64, f2=1, n2=32, f3=5)
    # srcnn_net.eval()
    # images_decode2 = srcnn_net(images_scaled)
    # print(f"images_decode2 shape: {images_decode2.shape}")

    # Display all plots
    plt.show()


if __name__ == '__main__':
    main()
