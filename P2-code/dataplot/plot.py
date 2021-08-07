import matplotlib.pyplot as plt
import numpy as np


def singlePlot(data,file=0,ylabel=0,xlabel=0):
    '''
    Simple plot of a 1-D array in indices of the array

    data   : The 1-D array to be plotted
    file   : name of file (with extension), if plot to be saved to a file
    ylabel : label of Y axis, defaulted to "Value"
    xlabel : label of X axis, defaulted to "Intensity"
    '''
    plt.plot(data)
    if xlabel==0:
        xlabel="Intensity"
    if ylabel==0:
        ylabel="Value"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if file == 0:
        plt.show()
    else:
        plt.savefig(file)
