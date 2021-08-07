import sys
import cv2 as cv
import numpy as np
from transform import localFourier as lft
from transform import mdc
from classification import lda,knn,kmean
from dataplot import plot as plt
from datetime import datetime as dt 

'''
Partition histopathological images to cellular and extra-cellular region
in order to detect deseases.

This is an implementation of partitionining method published in paper -

Partitioning Histopathological Images: An Integrated Framework for Supervised
Color-Texture Segmentation and Cell Splitting.

by Hui Kong*, Metin Gurcan, Senior Member, IEEE, and Kamel Belkacem-Boussaid, Senior Member, IEEE
'''

plft=[]  # global variable to store mean LFT features for positive samples
nlft=[]  # global variable to store mean LFT features for negative samples

def collectSampleLFT(sfile,stype):
    '''
    Calculates mean LFT features for the sample file name passed from input and
    stores the features in plft and nlft global arrays.
    Input: sfile -> name of the sample file
           stype -> 0 if positive sample : LFT data is added to array plft
                    1 if negative sample : LFT data is added to array nlft
    '''

    global plft,nlft                  # global variables to store mean LFT features for positive and negative samples
    
    img=cv.imread(sfile)              # convert file to image tensor
    if img is None:
        sys.exit("Could not read the image "+sfile)

    lfti=lft.getLFTmean(img)          # get 24 LFT features for each sample      
    if stype==0:
        if type(plft)==list:
            plft=np.array(lfti)       # add lft vectors from positive sample to plft
        else:
            plft=np.vstack([plft,lfti])
    else:
        if type(nlft)==list:
            nlft=np.array(lfti)       # add lft vectors from negative sample to nlft
        else:
            nlft=np.vstack([nlft,lfti])

'''
Main function

Usage : Modify following parameters -
          file : provide name of the prediction image (the histogram equalized image out of of histogramEqualisation.py)
          dir  : provide name of the directory where samples image patched are present
                 by default it searches in ..\samples\name of file excluding the extension
          n_samples : number of positive samples (should be same as number of negative samples)
          n_start   : sample number starting from (in case the numbering of the sampling are not starting from 1)
Assumptions:
          File extension is three characters long.
             - In case file extension is 4 chars, e.g. jpeg, modify file[:-4] to file[:-5] and file[-4:] to file[-5:] everywhere

'''
if __name__=="__main__":

#    file='43601_HE.bmp'
    file='national-cancer-institute_HE.bmp'
    dir="samples\\"+file[:-4]+"\\"
    
    # Calculate LFT features for all 11 x 11 sample training images

    n_samples=80
    n_start = 1
    
    for i in range(n_start,n_samples+n_start):

        sfile=dir+'PositiveSample'+str(i)+'.bmp'         # open positive sample bmp files from /samples directory 
        collectSampleLFT(sfile,0)                        # store positive sample lft features to plft array

        sfile=dir+'NegativeSample'+str(i)+'.bmp'         # open negative sample bmp files from /samples directory 
        collectSampleLFT(sfile,1)                        # store negative sample lft features to nlft array

    # Create integral map of the whole image

    img=cv.imread(file)                                  # convert file to image tensor
    if img is None:
        sys.exit("Could not read the image "+file)

    lfti=lft.getLFTIntegral(img)                         # get LFT features for all chanels in 24 integral maps

    # Iterative Fisher-Rao optimisation
    
    A,P,pltJ=lda.fisherRaoOptimization(plft,nlft,0.00001)# calculate optimised P (projection matrix) and A (RGB to MDC map) using iterative Fisher Rao optimisation 
    print("A*=",A)
    plt.singlePlot(pltJ,"PlotJ"+file[:-4]+".jpg","J","Iterations")                    # create plot of J(ratio of between-class spread / within-class spread) for each iteration
    mimg=mdc.RGB2MDC(img,A,file[:-4]+"_mdc"+file[-4:])   # store MDC image 

    # Identify 20 positive and negative cluster points using K-means

    plft=kmean.kmeans(plft)
    nlft=kmean.kmeans(nlft)

    # Segmentation using the learned machine

    X=np.vstack([plft,nlft])                             # data matrix for K-nn 
    y=np.hstack([np.ones(plft.shape[0],dtype=int),np.zeros(nlft.shape[0],dtype=int)]) # target vector for K-nn
    print("Getting mean lft:",dt.now())
    lftm=lft.getMeanlft(lfti)                            # calculate mean lft from 11 x 11 neighborhood of each input image pixel from lft integral map
    print("Starting knn:",dt.now())
    simg=knn.knn(X,y,lftm)                               # segment image using k-nn
    print("End knn:",dt.now())
    cv.imwrite(file[:-4]+"_segmented.bmp",simg)          # save segmented image (cells in green, outer-cell region in blue)

