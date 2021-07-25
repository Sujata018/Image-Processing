import sys
import cv2 as cv
import numpy as np
from lft import localFourier as lft
from classification import lda

'''
Partition histopathological images to cellular and extra-cellular region
in order to detect deseases.

This is an implementation of partitionining method published in paper -

Partitioning Histopathological Images: An Integrated Framework for Supervised
Color-Texture Segmentation and Cell Splitting.

by Hui Kong*, Metin Gurcan, Senior Member, IEEE, and Kamel Belkacem-Boussaid, Senior Member, IEEE
'''

if __name__=="__main__":

    file='43601_HE.bmp'
    dir="samples\\"+file[:-4]+"\\"
    
    # Calculate LFT features for all 11 x 11 sample training images

    for i in range(4):
        if i%2 ==0:
            sfile=dir+'PositiveSample'+str(i//2+1)+'.bmp'  # open bmp files from /samples directory
        else:
            sfile=dir+'NegativeSample'+str(i//2+1)+'.bmp'

        img=cv.imread(sfile)                               # convert file to image tensor
        if img is None:
            sys.exit("Could not read the image "+sfile)

        lfti=lft.getLFTmean(img)                           # get 24 LFT features for each sample      
        
        if i==0:
            plft=np.array(lfti)                            # add lft vectors from positive sample to plft
        elif i==1:
            nlft=np.array(lfti)                            # add lft vectors from negative sample to nlft
        elif i%2 ==0:
            plft=np.vstack([plft,lfti])
        else:
            nlft=np.vstack([nlft,lfti])

    # Create integral map of the whole image

    img=cv.imread(file)               # convert file to image tensor
    if img is None:
        sys.exit("Could not read the image "+file)

    lfti=lft.getLFTIntegral(img)       # get LFT features for all chanels in 24 integral maps
 
    A,P=lda.fisherRaoOptimization(plft,nlft)
    
