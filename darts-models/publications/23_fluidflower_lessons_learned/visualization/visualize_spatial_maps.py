#!/usr/bin/env python3

""""
Script to visualize the gas saturation and CO2 concentration
on an evenly spaced grid as required by the benchmark description
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def getFieldValues(fileName, nX, nY):
    print(f'Processing {fileName}.')

    csvData = np.genfromtxt(fileName, delimiter=',', skip_header=1)
    saturation = np.zeros([nX, nY])
    concentration = np.zeros([nX, nY])
    for i in np.arange(0, nY):
        saturation[:, i] = csvData[i*nX:(i+1)*nX, 2]
        concentration[:, i] = csvData[i*nX:(i+1)*nX, 3]

    return np.array(saturation), np.array(concentration)


def plotColorMesh(axs, x, y, z, i, title, limits=None):
    z = np.swapaxes(z, 0, 1)
    im = axs[i].pcolormesh(x, y, z, shading='flat', cmap='coolwarm')
    axs[i].axis([x.min(), x.max(), y.min(), y.max()])
    axs[i].axis('scaled')
    axs[i].set_title(title, fontsize=16)
    im.set_clim(limits)
    divider = make_axes_locatable(axs[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')


def visualizeSpatialMaps(output_dir, days=[1, 2, 3, 4, 5]):
    """Visualize a spatial map for the FluidFlower benchmark"""

    parser = argparse.ArgumentParser(
        description="This script visualizes the gas saturation and CO2 concentration "
                    "on an evenly spaced grid as required by the benchmark description."
    )
    parser.add_argument("-f", "--basefilename", default=output_dir + "/spatial_map",
                        help="The base name of the csv files to visualize. "
                             "Assumes that the files are named 'basefilename_<X>h.csv' "
                             "with X = 24, 48, ..., 120. Defaults to 'spatial_map'.")

    cmdArgs = vars(parser.parse_args())
    baseFileName = cmdArgs["basefilename"]

    xSpace = np.arange(0.0, 2.861, 1.0e-2)
    ySpace = np.arange(0.0, 1.231, 1.0e-2)
    x, y = np.meshgrid(xSpace, ySpace)

    nX = xSpace.size-1
    nY = ySpace.size-1

    n_plots = len(days)
    figS, axsS = plt.subplots(1, n_plots, figsize=(n_plots * 8, 8))
    figC, axsC = plt.subplots(1, n_plots, figsize=(n_plots * 8, 8))
    # figS = plt.figure(figsize=(18, 6))
    # figC = plt.figure(figsize=(18, 6))

    i = 0
    for day in days:
        time = '{:.0f}h'.format(day*24)
        fileName = baseFileName + '_' + time + '.csv'

        saturation, concentration = getFieldValues(fileName, nX, nY)

        timestamp = 't = {:.0f} day'.format(day) if day == 1 else 't = {:.0f} days'.format(day)
        plotColorMesh(axsS, x, y, saturation, i, 'saturation at ' + timestamp)
        plotColorMesh(axsC, x, y, concentration, i, 'concentration at ' + timestamp)
        i += 1

    figS.tight_layout()
    figS.savefig(f'{baseFileName}_saturation.png', bbox_inches='tight')
    figC.tight_layout()
    figC.savefig(f'{baseFileName}_concentration.png', bbox_inches='tight')
    print(f'Files {baseFileName}_saturation.png and {baseFileName}_concentration.png have been generated.')


def visualizeBoxC(output_dir, timestamps):
    fileName = output_dir + '/time_series_boxC.csv'

    xSpace = np.arange(1.13, 2.631, 1.0e-2)
    ySpace = np.arange(0.13, 0.431, 1.0e-2)
    x, y = np.meshgrid(xSpace, ySpace)

    nX = xSpace.size - 1
    nY = ySpace.size - 1

    n_plots = len(timestamps)
    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 8, 8))

    csvData = np.genfromtxt(fileName, delimiter=',')
    cells = csvData[0, 1:]

    i = 0
    for timestamp in timestamps:
        time = '{:.0f}h'.format(timestamp/3600)
        row = int(timestamp/600)

        concentration = np.zeros([nX, nY])

        for k in np.arange(nY):
            for j in np.arange(nX):
                concentration[j, k] = csvData[row, k*nX + j + 1]

        plotColorMesh(axs, x, y, concentration, i, 'concentration at ' + time)
        i += 1

    fig.tight_layout()
    fig.savefig(output_dir + '/boxC.png', bbox_inches='tight')


def visualizeDifferenceMaps(output_dir):
    """Visualize a spatial map for the FluidFlower benchmark"""

    parser = argparse.ArgumentParser(
        description="This script visualizes the gas saturation and CO2 concentration "
                    "on an evenly spaced grid as required by the benchmark description."
    )
    parser.add_argument("-f", "--basefilename", default=output_dir + "/spatial_map",
                        help="The base name of the csv files to visualize. "
                             "Assumes that the files are named 'basefilename_<X>h.csv' "
                             "with X = 24, 48, ..., 120. Defaults to 'spatial_map'.")

    cmdArgs = vars(parser.parse_args())
    baseFileName = cmdArgs["basefilename"]

    xSpace = np.arange(0.0, 2.861, 1.0e-2)
    ySpace = np.arange(0.0, 1.231, 1.0e-2)
    x, y = np.meshgrid(xSpace, ySpace)

    nX = xSpace.size - 1
    nY = ySpace.size - 1

    fig = plt.figure(figsize=(18, 6))

    fileName1 = f'{baseFileName}_{24}h.csv'
    fileName2 = f'{baseFileName}2_{24}h.csv'

    saturation2, concentration2 = getFieldValues(fileName2, nX, nY)
    saturation1, concentration1 = getFieldValues(fileName1, nX, nY)
    diff_sat, diff_conc = saturation1 - saturation2, concentration1 - concentration2

    sat_range = [0, 0.9]
    plotColorMesh(fig, x, y, saturation2, i=1, title='saturation, original', limits=sat_range)
    plotColorMesh(fig, x, y, saturation1, i=2, title='saturation, new', limits=sat_range)
    plotColorMesh(fig, x, y, diff_sat, i=3, title='saturation, difference')

    conc_range = [0, 1.9]
    plotColorMesh(fig, x, y, concentration2, i=4, title='concentration, original', limits=conc_range)
    plotColorMesh(fig, x, y, concentration1, i=5, title='concentration, new', limits=conc_range)
    plotColorMesh(fig, x, y, diff_conc, i=6, title='concentration, difference')

    fig.savefig(f'{baseFileName}_difference.png', bbox_inches='tight')
    print(f'File {baseFileName}_difference.png has been generated.')

if __name__ == "__main__":
    visualizeSpatialMaps(output_dir="../visualization", days=[1, 21])
    # visualizeDifferenceMaps(output_dir="../unstr_tpfa_refined")
