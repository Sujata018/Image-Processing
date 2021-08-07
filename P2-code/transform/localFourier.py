import sys
import cv2 as cv
import numpy as np


'''
This package contains techniques to calculate Local Fourier Transforms
'''

def addPadding(img,d=1):
    '''
    Add zero padding of d rows and columns to four sides of the image
    d = 1 by default

    Input : 2-D matrix m x n
    Output: 2-D matrix m+d x n+d
    '''
    row=img.shape[0]
    if len(img.shape)==2:   # image is 2D
        I0=np.c_[np.zeros((row,d),dtype=img.dtype),img,np.zeros((row,d),dtype=img.dtype)]
        col=I0.shape[1]
        I0=np.r_[np.zeros((d,col),dtype=I0.dtype),I0,np.zeros((d,col),dtype=I0.dtype)]
    elif len(img.shape)==3: # image is 3D
        hight=img.shape[2]
        Z=np.zeros((row,d,hight),dtype=img.dtype)
        I0=np.hstack([Z,img,Z])
        Z=np.zeros((d,I0.shape[1],hight),dtype=I0.dtype)
        I0=np.vstack([Z,I0,Z])
    else:
        sys.exit("image is not 2D or 3D")
    return I0

def removePadding(img,d=1):
    '''
    Remove padding of d rows and columns from four sides of the image, keeping depth intact
    d = 1 by default

    Input : 3-D matrix m+d x n+d x p
    Output: 3-D matrix m x n x p
    '''
    return img[d:-1*d,d:-1*d,:]

def addPaddingL(img,d=1,side='w'):
    '''
    Add zero padding of d columns to left of the image
    d = 1 by default

    Input : img  -> 3-D matrix m x n x p
            d    -> width of padding
            side -> 'w' : add padding to left of image, default = 'w'
                    'e' : add padding to right of image
                    'n' : add padding to top of image
                    's' : add padding to bottom of image
    Output: 3-D matrix m x d+n x p
    '''
    
    if len(img.shape)==3:             # image is 3D
        row,col,hight=img.shape       # get number of rows, cols and height of input image
        if side=='w':                 # add padding in north / south / east / west directions, as provided by side parameter
            I0=np.hstack([np.zeros((row,d,hight),dtype=img.dtype),img])
        elif side=='e':
            I0=np.hstack([img,np.zeros((row,d,hight),dtype=img.dtype)])
        elif side=='n':
            I0=np.vstack([np.zeros((d,col,hight),dtype=img.dtype),img])
        elif side=='s':
            I0=np.vstack([img,np.zeros((d,col,hight),dtype=img.dtype)])
        else:
            sys.exit("Invalid option for side of padding")
    else:
        sys.exit("image is not 3D")

    return I0

def getMeanlft(img):
    '''
    Get mean LFT from 11x11 neighborhood from each pixel of the integral map image

    Input : img  -> the integral map
    Output: mlft -> mean lfts
    Logic : shift integral map image in four diagonal directions by 5 pixels
            get sum by adding bottom-right + up-left - up-right - bottom-left
            get mean by dividing above lft map by 11*11=121
    '''
    d=5

    img=img.astype(np.float32)
    # Shift up-left 
    I01=addPaddingL(img,2*d,'e')  # add 2d padding to right 
    I01=addPaddingL(I01,2*d,'s')  # add 2d padding to bottom
    
    # Shift bottom-right 
    I02=addPaddingL(img,2*d,'w')  # add 2d padding to left
    I02=addPaddingL(I02,2*d,'n')  # add 2d padding to top

    # Shift up-right 
    I03=addPaddingL(img,2*d,'w')  # add 2d padding to left
    I03=addPaddingL(I03,2*d,'s')  # add 2d padding to bottom
    
    # Shift bottom-left 
    I04=addPaddingL(img,2*d,'e')  # add 2d padding to right
    I04=addPaddingL(I04,2*d,'n')  # add 2d padding to top

    Im=(I01-I03+I02-I04)/121      # mean = (sum of diagonal points - sum of cross diagonal points )/total number of pixels in the box
    
    oimg=removePadding(Im,d)      # remove paddings added
   
    return oimg                   # return image with 24 mean lfts for each pixel 

def shiftImage(img,d=1):
    '''
    This function creates 8 gray images by shifting the original image img
       with offset d (default d=1), in 8 directions, and returns simg, where
       simg[0] : I(0,-1)  left shift
       simg[1] : I(0,1)   right shift
       simg[2] : I(-1,0)  up shift
       simg[3] : I(1,0)   down shift
       simg[4] : I(-1,-1) up-left shift
       simg[5] : I(-1,1)  up-right shift
       simg[6] : I(1,-1)  down-left shift
       simg[7] : I(1,1)   down-right shift
       
       '''
    # Padding
    I0=addPadding(img)
    
    # Left shift
    I01=np.roll(I0,-1*d)
    
    # Right shift
    I02=np.roll(I0,d)

    # Up shift
    I03=np.roll(I0,-1*d,axis=0)

    # Down shift
    I04=np.roll(I0,d,axis=0)

    # Up-left shift
    I05=np.roll(I03,-1*d)

    # Up-right shift
    I06=np.roll(I03,d)

    # Down-left shift
    I07=np.roll(I04,-1*d)

    # Down-right shift
    I08=np.roll(I04,d)

    return np.dstack((I01,I02,I03,I04,I05,I06,I07,I08))


def computeLFTmap(simg):
    '''
    Compute LFT features based on 8-neighourhood of each pixel using following
    LFT maps of the shifted images:

    L1=I(0,-1)+I(0,1)+I(-1,0)+I(1,0)+I(-1,-1)+I(-1,1)+I(1,-1)+I(1,1)
    L2=[I(0,-1)+I(0,1)+I(-1,0)+I(1,0)]
        -[I(-1,-1)+I(-1,1)+I(1,-1)+I(1,1)]
    L3=0.707 x [I(-1,1)+I(1,1)-I(-1,-1)-I(1,-1)]
        +I(0,1)-I(0,-1)
    L4=0.707 x [I(1,-1)+I(1,1)-I(-1,-1)-I(-1,1)]
        +I(1,0)-I(-1,0)
    L5=I(0,-1)+I(0,1)-I(-1,0)-I(1,0)
    L6=I(-1,-1)+I(1,1)-I(-1,1)-I(1,1)
    L7=0.707 x [I(-1,-1)+I(1,-1)-I(-1,1)-I(1,1)]
        +I(0,1)-I(0,-1)
    L8=0.707 x [I(1,-1)+I(1,1)-I(-1,-1)-I(-1,1)]
        +I(-1,0)-I(1,0)
    '''

    # convert to signed float, as LFT features can be negative 
    simg=simg.astype(np.float32)

    # calculate LFT maps
    L1=np.sum(simg,axis=2)
    L2=simg[:,:,0]+simg[:,:,1]+simg[:,:,2]+simg[:,:,3]-simg[:,:,4]-simg[:,:,5]-simg[:,:,6]-simg[:,:,7]
    L3=0.707*(simg[:,:,5]+simg[:,:,7]-simg[:,:,4]-simg[:,:,6])+simg[:,:,1]-simg[:,:,0]
    L4=0.707*(simg[:,:,6]+simg[:,:,7]-simg[:,:,4]-simg[:,:,5])+simg[:,:,3]-simg[:,:,2]
    L5=simg[:,:,0]+simg[:,:,1]-simg[:,:,2]-simg[:,:,3]
    L6=simg[:,:,4]+simg[:,:,7]-simg[:,:,5]-simg[:,:,6]
    L7=0.707*(simg[:,:,4]+simg[:,:,6]-simg[:,:,5]-simg[:,:,7])+simg[:,:,1]-simg[:,:,0]
    L8=0.707*(simg[:,:,6]+simg[:,:,7]-simg[:,:,4]-simg[:,:,5])+simg[:,:,2]-simg[:,:,3]

    return np.dstack((L1,L2,L3,L4,L5,L6,L7,L8))
 
def createIntegralmap1(simg):
    '''
    Create integral map of a matrix or tensor
    integral map technique provides a very efficient way to compute the sum of an arbitrary
    subset of a 2-D matrix.
    The integral map at location (x,y) contains the sum of pixels above and to the left
    of (x,y), inclusive.

    This technique is an implementation of following paper :

    P. Viola and M. J. Jones, “Robust real-time face detection,”
    Int. J. Comput. Vis., vol. 57, no. 2, pp. 137–154, 2004
    '''
    print("image shape in input of integral map ", simg.shape)
    simg=simg.astype(np.float32)

    # Make padding layers to 0
    simg[0,:,:]=0
    simg[-1,:,:]=0
    simg[:,0,:]=0
    simg[:,-1,:]=0

    # For each channel of the 3-D image, compute integrals separately

    for depth in range(simg.shape[2]):
        for row in range(1,simg.shape[0]-1):
            for col in range(1,simg.shape[1]-1):
                simg[row,col,depth]=simg[row-1,col,depth]+simg[row,col-1,depth]+simg[row,col,depth]-simg[row-1,col-1,depth]
 
    return removePadding(simg)

def createIntegralmap(simg):
    '''
    Create integral map of a matrix or tensor
    integral map technique provides a very efficient way to compute the sum of an arbitrary
    subset of a 2-D matrix.
    The integral map at location (x,y) contains the sum of pixels above and to the left
    of (x,y), inclusive.

    This technique is an implementation of following paper :

    P. Viola and M. J. Jones, “Robust real-time face detection,”
    Int. J. Comput. Vis., vol. 57, no. 2, pp. 137–154, 2004
    '''
    simg=removePadding(simg)          # Remove padding layers
    iimg=cv.integral(simg)            # Create integrals map
 
    return iimg[1:,1:,:]

def getChanelLFTIntegral(img):
    '''
    Input : a gray image from a single chanel
    Output: a tensor with integral maps of 8 LFTs calculted for each pixel
            size : input row x input column x 8
    '''
    imglft=shiftImage(img)
    imglftmap=computeLFTmap(imglft)
    imglfti=createIntegralmap(imglftmap)

    return imglfti

def getLFTIntegral(img):
    '''
    Input : a 3-D color image 
    Output: a tensor with integral maps of 8 LFTs calculted for each chanel 
            size : input row x input column x 24
    '''
    b,g,r=img[:,:,0],img[:,:,1],img[:,:,2] # split into blue, green, red

    blft=getChanelLFTIntegral(b)       # get LFT features for each chanel in 8 integral maps
    glft=getChanelLFTIntegral(g)
    rlft=getChanelLFTIntegral(r)
    lfti=np.dstack([blft,glft,rlft])       # stack integral maps from all chanels: total 24 features per pixel

    return lfti

def getmeanlftgray(img):
    '''
    Get mean of LFT features over entire image.
    Input : A gray image
    Output: A vector with LFT features of the image with dimension 1 x 8  
    '''
    simg=shiftImage(img)                   # shift image in 8 directions
    slft=computeLFTmap(simg)               # compute lft map from shifted images
    lft=removePadding(slft,2)              # remove border pixels as those are invalid for LFT calculation and remove padding added due to image shifting
    mlft=np.mean(lft,axis=(0,1))           # get mean over entire image for each lft feature

    return mlft

def getLFTmean(img):
    '''
    Input : a 3-D color image 
    Output: a feature vector with 24 dimensions, one for each 8 LFTs in 3 chanels.
            Mean is calculted for entire image for 24 LFTs  
            size : 1 x 24
    '''
    b,g,r=img[:,:,0],img[:,:,1],img[:,:,2] # split into blue, green, red

    blft=getmeanlftgray(b)                 # get LFT features for each chanel in 8 integral maps
    glft=getmeanlftgray(g)
    rlft=getmeanlftgray(r)
    lfti=np.hstack([blft,glft,rlft])       # stack integral maps from all chanels: total 24 features per pixel

    return lfti
