#
# Filename: MBSRCNNTrain.py
#
# Description: Program that uses the classes in SuperResNetworks.py to
# #           instantiate and train a Multi-Branch SRCNN-style network for super resolution
# #           of 32x32 images uf ux up to 128x128.
#
# by Christopher J. Abel
#
# Revision History
# ----------------
#   05/03/2025 -- Original
#
# ------------------------------------------------------------
import math
import time
import pandas as pd
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
    variable1 = 'ux'
    variable2 = 'p'
    batch_size = 4
    imagepath1 = '../../../TestingSets/' + variable1 + '/'
    imagepath2 = '../../../TestingSets/' + variable2 + '/'
    parameter_file = "sr_net_conf2a.pth"

    # ----------------------------------------------------------------
    #
    #      Setup transform, datasets, dataloader, and
    #      primary network
    #
    transform1 = SuperResNetworks.ResizeTransform(32).transform
    dataset = SuperResNetworks.MBSRDataset(imagepath1, imagepath2, transform1=transform1, remove_alpha=False)
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images1_orig, images1_scaled, images2_orig, images2_scaled = next(iter(dataloader1))

    sr_net = SuperResNetworks.TwoBrSR(n0=4, f1=9, n1=128, f2=1, n2=64, f3=5)
    print(f"Loading previous parameter set: {parameter_file}")
    sr_net.load_state_dict(torch.load(parameter_file))
    sr_net.eval()

    # ------------------------------------------------------------
    #
    #       Process images through sr_net
    #
    images_decode = sr_net(images1_scaled, images2_scaled)

    figure1 = plt.figure()
    for i in range(0, 4):
        # Use numpy clip function to eliminate warning about values slightly
        # outside range ([0, 1]).  This is for Matplotlib image display only.
        image_orig = np.clip(images1_orig[i].permute(1, 2, 0), 0, 1)
        image1 = np.clip(images1_scaled[i].permute(1, 2, 0), 0, 1)
        # print(f"Decoded tensor: max = {torch.max(images_decode[i])}, min = {torch.min(images_decode[i])}, avg = {torch.mean(images_decode[i])}")
        image_decode = np.clip(images_decode[i].permute(1, 2, 0).detach().numpy(), 0, 1)
        print(
            f"image_decode mean = {np.mean(image_decode)}, max = {np.max(image_decode)}, min = {np.min(image_decode)}")
        figure1.add_subplot(3, 4, i + 1)
        plt.imshow(image_orig.squeeze())
        figure1.add_subplot(3, 4, i + 5)
        plt.imshow(image1.squeeze())
        figure1.add_subplot(3, 4, i + 9)
        plt.imshow(image_decode.squeeze())

    # Plot image 0 in a 1 x 3 row of images
    figure2 = plt.figure()
    image_orig = np.clip(images1_orig[0].permute(1, 2, 0), 0, 1)
    image1 = np.clip(images1_scaled[0].permute(1, 2, 0), 0, 1)
    image_decode = np.clip(images_decode[0].permute(1, 2, 0).detach().numpy(), 0, 1)
    figure2.add_subplot(1, 3, 1)
    plt.imshow(image_orig.squeeze())
    figure2.add_subplot(1, 3, 2)
    plt.imshow(image1.squeeze())
    figure2.add_subplot(1, 3, 3)
    plt.imshow(image_decode.squeeze())

    # Calculate and report PSNR for entire batch of testing images
    batch_size = 250
    dataloader2 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images1_orig, images1_scaled, images2_orig, images2_scaled = next(iter(dataloader2))
    images_decode = sr_net(images1_scaled, images2_scaled)
    images_decode = images_decode.detach()
    psnr_batch = PSNRCalc(images1_orig, images_decode)
    print(f"PSNR for the batch of images = {psnr_batch : .2f} dB")

    # Display plot of loss vs epoch
    plt.show()



if __name__ == '__main__':
    main()
