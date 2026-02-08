#!/usr/bin/env python3

""""
Script to visualize the time series quantities
as required by the benchmark description
"""

import numpy as np
import matplotlib.pyplot as plt


def visualizeTimeSeries():
    """Visualize time series for the FluidFlower benchmark"""

    p1 = [[]]
    p2 = [[]]
    gasA = [[]]
    disA = [[]]
    gasB = [[]]
    disB = [[]]

    # loop over csv files
    filenames = np.array(["str_coarse", "str_fine", "unstr_coarse", "unstr_fine"])
    for i, filename in enumerate(filenames):
        path = "../data/" + filename + "/time_series.csv"
        csvData = np.genfromtxt(path, delimiter=',', skip_header=1)
        t = csvData[:, 0] / 3600
        p1.append(csvData[:, 1])
        p2.append(csvData[:, 2])
        gasA.append((csvData[:, 3] + csvData[:, 4])*1e3)
        disA.append(csvData[:, 5]*1e3)
        gasB.append((csvData[:, 7] + csvData[:, 8])*1e3)
        disB.append(csvData[:, 9]*1e3)

    # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, p1[i+1][:], label=filename)
    plt.legend(loc='upper right')
    plt.title("Pressure at sensor S-1", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Pressure [Pa]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/P1.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, p2[i + 1][:], label=filename)
    plt.title("Pressure at sensor S-2", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Pressure [Pa]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/P2.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, gasA[i + 1][:], label=filename)
    plt.legend(loc='upper right')
    plt.title("Free gas $\mathregular{CO_2}$ in Box A", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Total mass [g]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/gasA.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, disA[i + 1][:], label=filename)
    plt.title("Dissolved $\mathregular{CO_2}$ in Box A", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Total mass [g]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/disA.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, gasB[i + 1][:], label=filename)
    plt.legend(loc='upper right')
    plt.title("Free gas $\mathregular{CO_2}$ in Box B", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Total mass [g]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/gasB.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    for i, filename in enumerate(filenames):
        plt.plot(t, disB[i + 1][:], label=filename)
    plt.title("Dissolved $\mathregular{CO_2}$ in Box B", fontsize=14)
    plt.xlabel("Time [h]", fontsize=12)
    plt.ylabel("Total mass [g]", fontsize=12)
    plt.tight_layout()
    fig.savefig('../data/disB.pdf', bbox_inches='tight')


if __name__ == "__main__":
    visualizeTimeSeries()
