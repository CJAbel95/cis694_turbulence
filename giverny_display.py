#
# Filename: giverny_display.py
#
# Description: Script to test the use of methods from the givernylocal
#             package to retrieve and plot data from the Johns Hopkins
#             Turbulence Database (JHTDB).
#
# by Christopher J. Abel
#    Javad M. Najafabadi
#
# Revision History
# ----------------
#   04/10/2025 -- Original
#
# ------------------------------------------------------------
import numpy as np
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
import matplotlib.pyplot as plt


def main():
    auth_token = 'edu.csuohio.vikes.s.mortazaviannajafabadi-38a671ff'
    dataset_title = 'channel'
    output_path = './giverny_output'
    variable = 'velocity'
    temporal_method = 'none'
    spatial_method = 'lag8'
    spatial_operator = 'field'

    # Instantiate dataset
    dataset = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=auth_token)

    # ############################################################
    #       2D Plane Demo
    # ############################################################
    time = 1.0
    nx = 65
    nz = 65
    n_points = nx * nz
    x_points = np.linspace(0.0, 0.5 * np.pi, nx, dtype=np.float64)
    print("x_points step = ", x_points[1] - x_points[0])
    print(x_points.size, x_points.shape)
    print(x_points[0], x_points[1], x_points[-1])
    y_points = 0.9
    z_points = np.linspace(0.0, 0.125 * np.pi, nz, dtype=np.float64)
    points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing='ij')],
                      dtype = np.float64).T
    result = getData(dataset, variable, time, temporal_method, spatial_method, spatial_operator, points)
    # set threshold for the number of numpy array elements to display.
    np.set_printoptions(threshold=10)

    print(f'num points = {len(points)}')
    print(f'\npoints = \n-\n{points}')
    # the 1st time index of result corresponds to the final time for the "position"
    # variable and the initial time for all other variables.
    print(f'\nresult (1st time index) = \n-\n{result[0]}\n')

    """
    Save result to TSV file
    """
    output_filename = 'turbulence_2D'
    write_interpolation_tsv_file(dataset, points, result, output_filename)

    """
    with result, generate a 2D contour plot.
        - a simple plot to quickly visualize the queried 2D plane.
    """;
    if nx >= 2 and nz >= 2:
        # user-defined plot parameters.
        # which time of the data to plot (0-based index, so the first time component is specified as 0).
        time_component = 0
        # which component (column) of the data to plot (0-based index, so the first component is specified as 0).
        plot_component = 0
        # figure width and height of the plot.
        figsize = 7
        # plot dpi.
        dpi = 67

        # calculate the strides to downscale the plot to avoid plotting more gridpoints than pixels.
        strides = np.ceil(np.array([nx, nz]) / (figsize * dpi)).astype(np.int32)

        # reformat 'result' for plotting.
        result_time = result[time_component]
        result_array = np.array(result_time)
        data_plot = np.zeros((nx, nz, len(result_time.columns)), dtype=np.float32)
        x_plot = np.zeros((nx, nz), dtype=np.float64)
        z_plot = np.zeros((nx, nz), dtype=np.float64)

        for i in range(nx):
            for j in range(nz):
                x_plot[i, j] = x_points[i]
                z_plot[i, j] = z_points[j]
                data_plot[i, j, :] = result_array[i * nz + j, :]

        # plot the data.
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        contour = plt.contourf(x_plot[::strides[0], ::strides[1]], z_plot[::strides[0], ::strides[1]],
                               data_plot[::strides[0], ::strides[1], plot_component],
                               levels=500, cmap='inferno')
        plt.gca().set_aspect('equal')
        # select the colorbar orientation depending on which axis is larger.
        colorbar_orientation = 'vertical' if (np.max(z_plot) - np.min(z_plot)) >= (
                    np.max(x_plot) - np.min(x_plot)) else 'horizontal'
        cbar = plt.colorbar(contour, shrink=0.67, orientation=colorbar_orientation)
        # rotate the horizontal colorbar labels.
        if colorbar_orientation == 'horizontal':
            for label in cbar.ax.get_xticklabels():
                label.set_rotation(90)
        cbar.set_label(f'{variable} {spatial_operator}', labelpad=12, fontsize=14)
        plt.title(f'{dataset_title}', fontsize=16)
        plt.xlabel('x', labelpad=7, fontsize=14)
        plt.ylabel('z', labelpad=7, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
