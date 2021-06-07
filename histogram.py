import sys
import matplotlib.pyplot as plt

import config
from readP5file import readfile

'''
Creates an 1-D histogram from the image matrix
Indices of the 1-D array represent each of 0 to L-1 intensity levels possible
Frequency of each level is stored in corresponding indices.
'''
def createHistogram():
    config.hist=[0 for _ in range(config.maxGray+1)] # initialise histogram size to MaxGray and set all frequencies to 0

    for i in range(config.rows):
        for j in range(config.columns):
            config.hist[config.A[i,j]] += 1

'''
Normalise a histogram
Inputs: X : 1-D array name
Output: normalised X
Logic: Divides each elements of X by sum of all elements
'''
def normalise(X):
    
    total=sum(X)
    return [x/total for x in X]

'''
Shows the plot of a histogram.
'''
def plotHistogram(data=0,file=0,ylabel=0):

    if data == 0:
        data=config.hist
    if ylabel==0:
        ylabel='# of pixels'

    plt.clf()              # clear previous plots
    plt.plot(data)
    plt.xlabel('Intensity')
    plt.ylabel(ylabel)

    if file == 0:
        plt.show()
    else:
        plt.savefig(file[:-4]+'_histogram.png')
