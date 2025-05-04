#
# Filename: PSNRTest.py
#
# Description: Program that uses the classes in SuperResNetworks.py to
#           instantiate and test an SRCNN-style network for super resolution
#           of 32x32 images uf ux up to 128x128.  This program calculates the PSNR
#           (peak SNR) metric between the original ground-truth image (128 x 128) and the upscaled
#           image from the 32 x 32 image.
#
# by Christopher J. Abel
#
# Revision History
# ----------------
#   04/19/2025 -- Original
#
# ------------------------------------------------------------
import math
import time
import SuperResNetworks
import os
import numpy as np
import torch
from torchvision.transforms import ToTensor, v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


def PSNRCalc(image_orig, image_sr):
    # Assumes images are PyTorch tensors scaled to a range of [0 : 1]
    #
    # N = total number of pixels
    N = image_orig.numel()
    mse = torch.mean((image_orig.float() - image_sr.float()) ** 2).item()
    psnr = 10 * math.log10(1 / mse)
    return psnr


def main():
    batch_size = 4
    variable = 'ux'
    # imagepath = '../Isotropic turbulence/training_slices/'
    imagepath = '../../TestingSets/' + variable + '/'
    parameter_file = "sr_net.pth"

    # Setup transform, dataset, dataloader, and encoder-decoder network
    transform1 = SuperResNetworks.SuperResTransform(32).transform
    dataset = SuperResNetworks.SuperResDataset(imagepath, transform1=transform1)
    print(f"There are {len(dataset)} images in {imagepath}")
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images_orig, images_scaled = next(iter(dataloader1))
    sr_net = SuperResNetworks.SRCNNmod(n0=4, f1=9, n1=64, f2=1, n2=32, f3=5)
    print(f"Loading previous parameter set: {parameter_file}")
    sr_net.load_state_dict(torch.load(parameter_file))
    sr_net.eval()

    # Process images through sr_net
    images_decode = sr_net(images_scaled)
    images_decode = images_decode.detach()

    # Compare shapes and printout a small number of pixels for the batch
    # of original images and the batch of super-resolved images.
    print(f"Shape of images_orig = {images_orig.shape}")
    print(f"Shape of images_decode = {images_decode.shape}")
    print(images_orig[0, 1, 50 : 52, 50 : 52])
    print(images_decode[0, 1, 50: 52, 50: 52])

    # Calculate PSNR for image 0
    psnr = PSNRCalc(images_orig[0], images_decode[0])
    print(f"PSNR for image 0 = {psnr : .2f} dB")


if __name__ == '__main__':
    main()
