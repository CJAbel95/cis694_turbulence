#
# Filename: SRCNNTest.py
#
# Description: Program that uses the classes in SuperResNetworks.py to
#           instantiate and test an SRCNN-style network for super resolution
#           of 32x32 images uf ux up to 128x128.
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


def main():
    batch_size = 4
    variable = 'ux'
    # imagepath = '../Isotropic turbulence/training_slices/'
    imagepath = '../../TestingSets/' + variable + '/'
    parameter_file = "sr_net.pth"

    # Setup transform, dataset, dataloader, and encoder-decoder network
    transform1 = SuperResNetworks.ResizeTransform(32).transform
    dataset = SuperResNetworks.SuperResDataset(imagepath, transform1=transform1, remove_alpha=False)
    print(f"There are {len(dataset)} images in {imagepath}")
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images_orig, images_scaled = next(iter(dataloader1))
    sr_net = SuperResNetworks.SRCNNmod(n0=4, f1=9, n1=64, f2=1, n2=32, f3=5)
    # sr_net = SuperResNetworks.SRCNNmodBn(n0=4, f1=9, n1=64, f2=5, n2=32, f3=5)
    # sr_net = SuperResNetworks.SRCNNmod2(n0=4, f0=3, f1=9, n1=64, f2=1, n2=32, f3=5)
    # sr_net = SuperResNetworks.SRCNNmodRl(n0=4, f1=9, n1=64, f2=1, n2=32, f3=5)
    print(f"Loading previous parameter set: {parameter_file}")
    sr_net.load_state_dict(torch.load(parameter_file))
    sr_net.eval()

    # Process images through sr_net
    images_decode = sr_net(images_scaled)
    figure1 = plt.figure()
    for i in range(0, 4):
        # Use numpy clip function to eliminate warning about values slightly
        # outside range ([0, 1]).  This is for Matplotlib image display only.
        image_orig = np.clip(images_orig[i].permute(1, 2, 0), 0, 1)
        image1 = np.clip(images_scaled[i].permute(1, 2, 0), 0, 1)
        # print(f"Decoded tensor: max = {torch.max(images_decode[i])}, min = {torch.min(images_decode[i])}, avg = {torch.mean(images_decode[i])}")
        image_decode = np.clip(images_decode[i].permute(1, 2, 0).detach().numpy(), 0, 1)
        print(f"image_decode mean = {np.mean(image_decode)}, max = {np.max(image_decode)}, min = {np.min(image_decode)}")
        figure1.add_subplot(3, 4, i + 1)
        plt.imshow(image_orig.squeeze())
        figure1.add_subplot(3, 4, i + 5)
        plt.imshow(image1.squeeze())
        figure1.add_subplot(3, 4, i + 9)
        plt.imshow(image_decode.squeeze())

    # Plot image 0 in a 1 x 3 row of images
    figure2 = plt.figure()
    image_orig = np.clip(images_orig[0].permute(1, 2, 0), 0, 1)
    image1 = np.clip(images_scaled[0].permute(1, 2, 0), 0, 1)
    image_decode = np.clip(images_decode[0].permute(1, 2, 0).detach().numpy(), 0, 1)
    figure2.add_subplot(1, 3, 1)
    plt.imshow(image_orig.squeeze())
    figure2.add_subplot(1, 3, 2)
    plt.imshow(image1.squeeze())
    figure2.add_subplot(1, 3, 3)
    plt.imshow(image_decode.squeeze())

    # Display all Matplotlib figures
    plt.show()


if __name__ == '__main__':
    main()
