#
# Filename: SRTraining.py
#
# Description: Program that uses the classes in SuperResNetworks.py to
#           instantiate and train a U-Net style network for super resolution
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
    plot_loss = True
    num_epochs = 1000
    batch_size = 4
    learning_rate = 6e-10
    imagepath = '../Isotropic turbulence/training_slices/'
    parameter_file = "sr_net.pth"

    # Setup transform, dataset, dataloader, and encoder-decoder network
    transform1 = SuperResNetworks.SuperResTransform(32).transform
    dataset = SuperResNetworks.SuperResDataset(imagepath, transform1=transform1)
    print(f"There are {len(dataset)} images in {imagepath}")
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sr_net = SuperResNetworks.BaseSRNet()
    if os.path.exists(parameter_file):
        print(f"Loading previous parameter set: {parameter_file}")
        sr_net.load_state_dict(torch.load(parameter_file))
        sr_net.train()
    else:
        print(f"Starting training with random initial parameter values.")
        sr_net.apply(SuperResNetworks.initialize_weights)
        sr_net.train()

    # Setup loss function and optimizer
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(sr_net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(sr_net.parameters(), lr=learning_rate)

    #
    # Training loop
    #
    loss_v_epoch = []
    t_start = time.time()  # Capture wall clock time at start of training
    for epoch in range(0, num_epochs):
        for batch_idx, (img_orig, img) in enumerate(dataloader1):
            img_out = sr_net(img)
            loss = loss_function(img_orig, img_out)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Print Epoch # and loss for every 10 epochs
        loss_v_epoch.append([epoch, loss.item()])
        if (epoch % 10 == 0): print(f"Epoch: {epoch} \tLoss = {loss.item()}")

    #
    # Results. Save model parameters in parameter_file.
    #
    torch.save(sr_net.state_dict(), parameter_file)
    t_end = time.time()  # Wall clock time after training is complete
    minutes = (t_end - t_start) // 60
    seconds = t_end - t_start - 60 * minutes
    # Print out final loss value
    print(f"\nTraining over {num_epochs} epochs took {minutes:.0f} minutes and {seconds:.0f} seconds.")
    print(f"After training, loss = {loss_v_epoch[-1][1]:.4f}, RMS loss = {math.sqrt(loss_v_epoch[-1][1]):.4f}")


if __name__ == '__main__':
    main()
