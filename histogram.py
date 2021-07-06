import sys
import matplotlib.pyplot as plt
import numpy as np

import config

'''
Creates an 1-D histogram from the image matrix
Indices of the 1-D array represent each of 0 to L-1 intensity levels possible
Frequency of each level is stored in corresponding indices.

Input: img - a 2-D gray image. If no image is passed, default is taken from config.A
                       
'''
def createHistogram(img=0):
    if type(img) == np.ndarray:
        config.maxGray=255
        config.A=img.astype(int)
        shape=img.shape
        config.rows=shape[0]
        config.columns=shape[1]
          
    config.hist=[0 for _ in range(config.maxGray+1)] # initialise histogram size to MaxGray and set all frequencies to 0

    for i in range(config.rows):
        for j in range(config.columns):
            config.hist[config.A[i,j]] += 1

    if type(img) == np.ndarray:
        return config.hist

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
Inputs : data : A numpy array with number of pixels for intensities populated where intensity value=index of the array.
                Default data: config.hist
                In case data is a list, a joint plot is created (this functionality can be used to show original and equalized histograms side by side)
         file : Output file name to store the plot
                Default : plot is shown, not saved
         ylabel : To be used for plotting other similar data apart from histogram
                Default : Intnsity
'''
def plotHistogram(data=0,file=0,ylabel=0):

    if data == 0:
        data=config.hist
    if ylabel==0:
        ylabel='# of pixels'

    plt.clf()              # clear previous plots
    if type(data)==list:
        plt.subplot(121)
        plt.plot(data[0])
        plt.title('Original')
        plt.xlabel('Intensity')
        plt.ylabel(ylabel)

        plt.subplot(122)
        plt.plot(data[1])
        plt.title('Equalized')
        plt.xlabel('Intensity')
        plt.ylabel(ylabel)

    else:
        plt.plot(data)
        plt.xlabel('Intensity')
        plt.ylabel(ylabel)


    if file == 0:
        plt.show()
    else:
        plt.savefig(file[:-4]+'_histogram.png')
        
'''
Does global histogram equalizetion using formula g(x,y)=255*(f(x,y)-Lmin)/(Lmax-Lmin)
'''
def equalizeHistogramLinear(greyImg):
    minGray=np.min(greyImg)
    config.A87=(greyImg-minGray)/(np.max(greyImg)-minGray)*255
    return config.A87

'''
Does global histogram equalizetion using formula g(x,y)=255*(f(x,y)-Lmin)^2/(Lmax-Lmin)
'''
def equalizeHistogramSquare(greyImg):
    minGray=np.min(greyImg)
    maxGray=np.max(greyImg)
    imgNorm=(greyImg-minGray)/maxGray
    imgSquare=np.square(imgNorm)
    config.A876=imgSquare/np.max(imgSquare)*255
    return config.A876

'''
Does global histogram equalizetion using formula g(x,y)=255*(f(x,y)-Lmin)^0.5/(Lmax-Lmin)
'''
def equalizeHistogramRoot(greyImg):
    minGray=np.min(greyImg)
    maxGray=np.max(greyImg)
    imgSqrt=np.sqrt(greyImg-minGray)
    imgSqrt=imgSqrt.astype(int)
    config.A876=imgSqrt/np.max(imgSqrt)*255
    return config.A876 

'''
Equalise histogram with transform function 1-1/(1+6x^2)
'''
def equalizeHistogram1plusxsq(greyImg):
    minGray=np.min(greyImg)
    maxGray=np.max(greyImg)
    imgNorm=(greyImg-minGray)/maxGray
    img=1-(1/(np.square(6*imgNorm)+1))
    config.A876=img/np.max(img)*255
    return config.A876 

'''
Equalise histogram - creates a flat histogram
'''
def equalizeHistogramFlat(greyImg):

    imgran=greyImg+np.random.uniform(-1,+1,greyImg.shape) # Add random noise
    imgflat=imgran.flatten()                              # flatten the matrix to 1-D for transformation
    imgsort=np.argsort(imgflat)                           # get sorted order of each intensity
    bin=len(imgsort)//255                                 # set bin size for histogram as total # of pixels/total # of different intensity levels 
    for i in range(len(imgsort)):                         # spread intensity of all pixels in order of their rank in the sorted array from 0 to 255
        imgflat[imgsort[i]]=i//bin
    imgspectrum=imgflat.reshape(greyImg.shape)
    return imgspectrum
