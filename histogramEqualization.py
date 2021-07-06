import sys
import cv2 as cv
import numpy as np
import config
from histogram import createHistogram,plotHistogram,equalizeHistogramRoot,equalizeHistogram1plusxsq,equalizeHistogramFlat

'''
To increase contrast, perform histogram equalization.

command line argument : file name of a grey level image

'''

if __name__=='__main__':
    if len(sys.argv)!=2:
        sys.exit("Usage ",sys.argv[0]," <file>")

    config.initialise()            # internal storage initialisation

    file=sys.argv[1]

    img=cv.imread(file,0)          # convert file to grey level matrix

    if img is None:
        sys.exit("Could not read the image "+file)


    equ=equalizeHistogramFlat(img) # equalize histogram - best one , totally flat histogram
    eq2=equalizeHistogramRoot(img) # equalise histogram - square root transform function
    eq1=cv.equalizeHist(img)       # equalise histogram - using open cv
    eq3=equalizeHistogram1plusxsq(img)        # equalise histogram - using transform function 1-(1/(1+6.x^2))

    cv.imwrite(file[:-4]+'_HE1'+file[-4:],equ)# store output images names extended to _HE
    cv.imwrite(file[:-4]+'_HE'+file[-4:],eq1) 
    cv.imwrite(file[:-4]+'_HE2'+file[-4:],eq2) 
    cv.imwrite(file[:-4]+'_HE3'+file[-4:],eq3) 

    origHist=createHistogram(img)             # plot histogram of original image
    HE=createHistogram(eq1)                   # plot histogram of openCV HE
    plotHistogram([origHist,HE],file[:-4]+'_HE'+file[-4:]) # save histogram comparisons

    HE1=createHistogram(equ)                  # plot histogram after equalization
    plotHistogram([origHist,HE1],file[:-4]+'_HE1'+file[-4:])

    HE2=createHistogram(eq2)                  # plot histogram after square root transform
    plotHistogram([origHist,HE2],file[:-4]+'_HE2'+file[-4:])

    HE3=createHistogram(eq3)                  # plot histogram after square root transform
    plotHistogram([origHist,HE3],file[:-4]+'_HE3'+file[-4:])
