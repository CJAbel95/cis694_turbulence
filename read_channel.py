#
# Filename: read_channel.py
#
# Description: Python program to read data from a cutout
#               from the JHTDB.
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


def main():
    filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\'
                'CIS 694 -- Deep Learning\\Assignments\\Final Project\\cutouts\\')

    # filename = 'channel_x2048\\channel.h5'
    filename = 'channel_z1536\\channel.h5'
    with h5py.File(filepath + filename, 'r') as f:
        print('Keys: %s' % f.keys())
        dataset_name = 'Velocity_0001'
        data = f[dataset_name][:]
        xdata = f['zcoor'][:]

    print("xcoor = ", xdata)

    reshaped_data = np.transpose(data, (3, 0, 1, 2))
    # Plot the first slice of each dimension
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    print("reshaped_data shape = ", reshaped_data.shape)
    print(reshaped_data[0][0])

    # for i in range(3):
    #     im = axes[i].imshow(reshaped_data[i][0], cmap='viridis')  # Plotting the first slice of each reshaped dimension
    #     axes[i].set_title(f'Data slice {i + 1}')
    #     fig.colorbar(im, ax=axes[i])
    #
    # plt.show()



if __name__ == '__main__':
    main()
