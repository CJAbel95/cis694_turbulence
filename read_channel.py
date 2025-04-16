#
# Filename: read_channel.py
#
# Description: Python program to read data from a cutout
#               obtained from the JHTDB -- https://turbulence.idies.jhu.edu/cutout/jobs
#
# by Christopher J. Abel
#
# Revision History
# ----------------
#   04/13/2025 -- Original
#
# ------------------------------------------------------------
import h5py
from lxml import etree
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mplt
import pandas as pd


def main():
    filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\'
                'CIS 694 -- Deep Learning\\Assignments\\Final Project\\cutouts\\')

    # filename = 'channel_x2048\\channel.h5'
    filename = 'channel5200\\channel5200.h5'
    with h5py.File(filepath + filename, 'r') as f:
        print('Keys: %s' % f.keys())
        dataset_name = 'Velocity_0002'
        result = f[dataset_name][:]
        result_array = np.array(result)
        print(f"result_array shape = {result_array.shape}")
        x_points = f['xcoor'][:]
        y_points = f['ycoor'][:]
        z_points = f['zcoor'][:]
        xyz_points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing='ij')],
                              dtype=np.float64).T
        print(f"xyz_points shape = {xyz_points.shape}")
        u_field = result_array[0, :, :, 0]

        fig1, axes1 = plt.subplots(1, 2)
        axes1[0].imshow(u_field, cmap='seismic', vmin=-3, vmax=3)
        img = mplt.image.imread('../MiscellaneousCode/slice_t1.000.png')
        axes1[1].imshow(img)
        plt.show()

    # reshaped_data = np.transpose(data, (3, 0, 1, 2))
    # Plot the first slice of each dimension
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # print("reshaped_data shape = ", reshaped_data.shape)
    # print(reshaped_data[0][0])

    # for i in range(3):
    #     im = axes[i].imshow(reshaped_data[i][0], cmap='viridis')  # Plotting the first slice of each reshaped dimension
    #     axes[i].set_title(f'Data slice {i + 1}')
    #     fig.colorbar(im, ax=axes[i])
    #
    # plt.show()



if __name__ == '__main__':
    main()
