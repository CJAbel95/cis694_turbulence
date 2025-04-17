#
# Filename: examine_jhtdb_data.py
#
# Description: Analyzing structure and values of data for one
#               time point in the iso1024coarse dataset from JHTDB.
#
#
# by Christopher J. Abel
#
# Revision History
# ----------------
#   04/15/2025 -- Original
#
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    filename = ("../Isotropic turbulence"
                "/training_slices/Iso1024CoarseVelGradient/isotropic1024coarse_t0.500.csv")
    # df_pressure = pd.read_table(filename, skiprows=1) # For .tsv files
    dataframe = pd.read_csv(filename, skiprows=0)      # For .csv files
    print(f"dataframe keys: {dataframe.keys()}\n")
    print("First 10 rows:", dataframe.loc[:9, ['x', 'y', 'z', 'duxdx', 'duxdy']])

    print("\nSummary Statistics:")
    print(dataframe.describe().loc[:, ['duxdx', 'duxdy']])

    print("\nPercentage of points within specified limits")
    count = dataframe['duxdx'].count()
    limits = [3, 5, 10, 15, 20]
    print('vmin/vmax\tduxdx(%)\tduxdy(%)')
    print('---------\t--------\t--------')
    for limit in limits:
        duxdx_percentage = len(dataframe[(dataframe['duxdx'] >= -limit) & (dataframe['duxdx'] <= limit)])/count * 100
        duxdy_percentage = len(dataframe[(dataframe['duxdy'] >= -limit) & (dataframe['duxdy'] <= limit)]) / count * 100
        print(f"{-limit:4}/{limit:4}\t   {duxdx_percentage:.1f}%\t{duxdy_percentage:.1f}%")

    fig1, ax1 = plt.subplots()
    ax1.hist(dataframe['duxdx'], bins=50)
    ax1.set_title('Histogram of duxdx')
    plt.xlabel('duxdx')
    plt.ylabel('frequency')
    plt.yscale('log')

    fig2, ax2 = plt.subplots()
    ax2.hist(dataframe['duxdy'], bins=50)
    ax2.set_title('Histogram of duxdy')
    plt.xlabel('duxdy')
    plt.ylabel('frequency')
    plt.yscale('log')

    plt.show()


if __name__ == '__main__':
    main()
