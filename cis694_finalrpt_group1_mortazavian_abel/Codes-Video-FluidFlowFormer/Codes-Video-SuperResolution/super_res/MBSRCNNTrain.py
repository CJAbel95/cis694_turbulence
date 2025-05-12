#
# Filename: MBSRCNNTrain.py
#
# Description: Program that uses the classes in SuperResNetworks.py to
# #           instantiate and train a Multi-Branch SRCNN-style network for super resolution
# #           of 32x32 images of ux up to 128x128.
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

def main():
    variable1 = 'ux'
    variable2 = 'p'
    plot_loss = True
    num_epochs = 100
    batch_size = 8
    learning_rate = 5e-4
    # imagepath = '../Isotropic turbulence/training_slices/'
    imagepath1 = '../../../TrainingSets/' + variable1 + '/'
    imagepath2 = '../../../TrainingSets/' + variable2 + '/'
    parameter_file = "sr_net.pth"
    parameter_best_file = "sr_net_best.pth"
    loss_csv_file = "loss_v_epoch.csv"
    restart = True
    use_cpu = False

    # Change device to CUDA, if available
    if torch.cuda.is_available() and not use_cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ----------------------------------------------------------------
    #
    #      Setup transform, datasets, dataloader, and
    #      primary network
    #
    transform0 = SuperResNetworks.AugmentTransform().transform
    transform1 = SuperResNetworks.ResizeTransform(32).transform
    # transform1 = SuperResNetworks.ResizeFiltTransform(32).transform
    dataset = SuperResNetworks.MBSRDataset(imagepath1, imagepath2, transform_orig=transform0, transform1=transform1,
                                               remove_alpha=False)
    print(f"There are {len(dataset)} images in {imagepath1}")
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images1_orig, images1_scaled, images2_orig, images2_scaled = next(iter(dataloader1))
    print(f"{images1_orig.shape}, {images1_scaled.shape}, {images2_orig.shape}, {images2_scaled.shape}")

    # Move CNN to CUDA, if available, otherwise CPU
    sr_net = SuperResNetworks.TwoBrSR(n0=4, f1=9, n1=128, f2=1, n2=64, f3=5).to(device)

    # Load prior parameter file, if it exists
    if os.path.exists(parameter_file) and not restart:
        print(f"Loading previous parameter set: {parameter_file}")
        sr_net.load_state_dict(torch.load(parameter_file))
        sr_net.train()
    else:
        print(f"Starting training with random initial parameter values.")
        sr_net.apply(SuperResNetworks.initialize_weights)
        sr_net.train()

    # Setup loss function and optimizer
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(sr_net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(sr_net.parameters(), lr=learning_rate)

    # ------------------------------------------------------------
    #
    #       Training loop
    #

    best_loss = 10.0
    loss_v_epoch = []
    t_start = time.time()  # Capture wall clock time at start of training
    for epoch in range(0, num_epochs):
        epoch_loss = 0.0
        #
        # Capture next batch of images from dataloader. The tuple of image batches:
        #       1. img1_orig: Batch of Ground-Truth HR images of primary variable
        #       2. img1: Batch of downsampled versions of img1_orig: fed to branch 1 of CNN
        #       3. img2_orig: Batch of full-scale images of secondary variable
        #       4. img2: Batch of downsampled versions of img2_orig: fed to branch 2 of CNN
        #
        for batch_idx, (img1_orig, img1, img2_orig, img2) in enumerate(dataloader1):
            img1_orig, img1 = img1_orig.to(device), img1.to(device)
            img2 = img2.to(device)
            img_out = sr_net(img1, img2)
            # print(f"{img1_orig.shape}, {img1.shape}, {img2.shape}, {img_out.shape}")
            loss = loss_function(img1_orig, img_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()
            epoch_loss += loss.item()

        # Print Epoch # and loss for every 10 epochs
        # If epoch loss reaches a new local minimum, save "best"
        # parameter set.
        epoch_loss = epoch_loss / len(dataloader1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(sr_net.state_dict(), parameter_best_file)
        loss_v_epoch.append([epoch, epoch_loss])
        if (epoch % 5 == 0): print(f"Epoch: {epoch} \tLoss = {loss.item()}")

    # Save loss vs epoch to .csv file
    df1 = pd.DataFrame(loss_v_epoch, columns=['Epoch', 'Loss'])
    df1.to_csv(loss_csv_file, index=False)

    # Plot loss vs epoch
    lve_arr = np.array(loss_v_epoch)
    fig1, ax1 = plt.subplots()
    ax1.plot(lve_arr[:, 0], lve_arr[:, 1], 'ro-')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)

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

    # Display plot of loss vs epoch
    plt.show()



if __name__ == '__main__':
    main()
