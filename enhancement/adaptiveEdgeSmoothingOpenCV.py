import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
Program for adaptive edge smoothing.

Creates mean map based on a given Box Filter, and E[x^2] map.
Calculates variance = E[x^2]-(E[x])^2
Gets the mean with the minimum variance, inside the kernel window.
This process preserves edges, as variance is maximum for edge regions.

Parameters : files -> list containing the inout file names
             ksize -> width (in pixel) of the square filter for smoothing
'''
if __name__=='__main__':

    files=['balloons_noisy.ascii.pgm','Pegeon.PNG']    # List of input files
    ksize=5                                            # Size of the box kernel

    for i in range(len(files)):                        # For each input file
        img=cv.imread(files[i],0)                      #  Read input
        meanMap=cv.blur(img,(ksize,ksize))             #  Calculate mean based on the box kernel for each pixel positions
        cv.imwrite(files[i][:-4]+'_blur.jpg',meanMap)  

        sqr=np.square(img,dtype=np.uint16)              
        meanSqMap=cv.blur(sqr,(ksize,ksize))           # Calculate E[x^2] 

        varianceMap=meanSqMap-np.square(meanMap,dtype=np.uint16) # Variance = E[x^2] - (E[x])^2

        pad=ksize//2 
      
        rows,cols=img.shape
        oimg=np.zeros((rows,cols),dtype=np.uint8)      # initialise output image as all 0
        
        ## For each pixel of the variance map, calculate window size as the portion of the image, if the center pixel
        ## of the kernel is kept on top of the variance map. Get the position of minimum variances within the window,
        ## value from the mean map from that position is copied to output.
        

        for row in range(rows):                        
            startRow=max(0,row-pad)                    # start row of the window. In case the kernel crosses the input image boundary, only the portion inside the input image in considered in window.
            endRow=min(rows,row+pad+1)                 # end row+1 of the window is calculated considering no spill beyong border
            for col in range(cols):
                startCol=max(0,col-pad)                # start column of window
                endCol=min(cols,col+pad+1)             # end column of window
                r,c=np.unravel_index(np.argmin(varianceMap[startRow:endRow,startCol:endCol]),(endRow-startRow,endCol-startCol))
                outRow=startRow+r                      # get position of the minimum variance (within window), and convert it to the position in input image
                outCol=startCol+c
                oimg[row,col]=meanMap[outRow,outCol]   # get value of the calculated position from mean map to output image
        cv.imwrite(files[i][:-4]+'_AdaptiveEdge.jpg',oimg)

        ## Plot original image, smoothed image with the box filter and adaptive edge smoothed image in a row, and save the plot
        plt.clf()

        plt.subplot(131)
        plt.imshow(img,cmap='Greys_r')
        plt.title('Original')
        
        plt.subplot(132)
        plt.imshow(meanMap,cmap='Greys_r')
        plt.title('Blur')
 
        plt.subplot(133)
        plt.imshow(oimg,cmap='Greys_r')
        plt.title('Adaptive Edge Blur')

        plt.savefig(files[i][:-4]+'_AdaptiveEdgeResults.jpg')
                
                
        
