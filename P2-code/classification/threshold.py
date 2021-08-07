import sys
import numpy as np

'''
This package contains classification techniques based on global thresolding

NOTE : This code is not used from the main program, as individual pixel processing consumed tremendous time.
       Since background white regions are taken as negative samples during test data collection process,
       fast matrix manipulation of numpy is utilised by processing the whole image together, which completed
       within minutes.
'''

def separateWhite(rgbimg,lftimg,thresold):
    '''
    Checks all pixels in the rgb image.
    If sun of the rgb values > threshold, assigns membership of the pixel as background, and colors in blue.
    Otherwise, calculates mean of the LFT features from 11 x 11 neighborhood of the corresponding pixel from lft integral image
               and stores it in output matrix for feeding to a classification process.

    Inputs : rgbimg   -> RGB image
             lftimg   -> LFT integral image, # of rows and columns same as rgbimg 
             thresold -> a scalar value to be used as thresold (between 0 to 255 * 3
    
    Outputs : simg -> image with same dimensions as rgbimg, where all pixels larger than the threshold, are marked in blue as background pixel
                      and other pixels are all 1 (dummy value)
              z    -> a matrix with 24 LFT features at its columns and each pixel for prediction in its rows.
                      this matrix is intended to be an input to a classification process for prediction
    '''
    print("rgb shape ",rgbimg.shape, " LFT shape ", lftimg.shape)
    row,col=rgbimg.shape[:-1]                  # Get number of rows and columns in the image 
    if lftimg.shape[:-1]!=(row,col):           # If 2-D dimension of rgb and mdc images are not matching, throw error 
        sys.exit("Row and columns of rgb and mdc images are not matching.")

    z=[]                                       # initialise prediction matrix  
    simg=np.ones(rgbimg.shape,dtype=int)       # create output image with all ones
    for i in range(row):                       # check all pixels in rgbimg, 
        for j in range(col):
            if sum(rgbimg[i,j,:])>= thresold:  # if r+g+b >= thresold, color output in blue
                simg[i,j,:]=(255,0,0)
            else:                              # otherwise add to z for prediction
                if 4<i<(row-5) and 4<j<(col-5):# get mean LFT from 11 x 11 neighorhood from integral map
                    lft=lftimg[i+5,j+5,:]+lftimg[i-5,j-5,:]-lftimg[i+5,j-5,:]-lftimg[i-5,j+5,:] 
                else:                          # if the pixel is at wide 5 border, it can not have 11 x 11 neighborhood, so take direct LFT 
                    lft=lftimg[i,j,:]
                if type(z)== list:
                    z=lft           
                else:
                    z=np.vstack([z,lft])
                
    return simg,z                             # return image with whitese background colored in blue, and the matrix of lft values of other pixels for prediction
