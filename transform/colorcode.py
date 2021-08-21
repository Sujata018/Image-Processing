import sys
import numpy as np

'''
This module contains methods related to color code conversion of color images.
'''

def RGB2XYZ(imgRGB):
    '''
    Normalise RGB colors
    X=X/(X+Y+Z),
    Y=Y/(X+Y+Z),
    Z=Z/(X+Y+Z),
    where X,Y,Z are the RGB channel intensities of a pixel
    '''

    s=np.sum(imgRGB,axis=2)          # calculate r+g+b for each pixel of the image
    s=np.dstack([s,s,s])             # stack to 3-D image for element-wise division
    k=np.divide(imgRGB,s,where=s!=0) # divide rgb values in input by sum of r,g,b gray values for that pixel
    return k
    
def RGB2CMY(imgRGB):
    '''
    Convert RGB to CMY
    C = 1 - R
    M = 1 - G
    Y = 1 - B,
    where R=r/255, G=g/255, B=b/255, r,g,b are the gray values of a pixel in 3 channels.
    
    '''
   
    return 1-(imgRGB/255) # Normalize RGB image from 0-255 to 0-1 range, then subtract from 1 to get CMY

def CMY2RGB(imgCMY):
    '''
    Convert CMY to RGB
    R = 1 - C
    G = 1 - M
    B = 1 - Y,
    '''
   
    return 1-imgCMY 

def CMY2CMYK(imgCMY):
    '''
    Convert CMY to CMYK
    K = min(C,M,Y),
    if K=1, C=M=Y=0
    else, C=(C-K)/(1-K), M=(M-K)/(1-K), Y=(Y-K)/(1-K)
    '''

    K=np.amin(imgCMY,axis=2)              # minimum of C,M,Y is to be mixed to Black(K)
    KKK=np.dstack([K,K,K])                # make black matrix(K) 3-D so it can be subtracted from CMY
    CMY=imgCMY-KKK                        # if K=1, C,M,Y=0, else C=C-K, M=M-K, Y=Y-K
    K1=np.ones(CMY.shape)-KKK             # 1-K 3-D matrix for division from CMY
    np.divide(CMY,K1,out=CMY,where=K1!=0) # if K=1, C,M,Y=0, else C=(C-K)/(1-K), M=(M-K)/(1-K), Y=(Y-K)/(1-K)
    return np.dstack((CMY,K))             # concatenate K matrix after CMY, to form CMYK

def CMYK2CMY(imgCMYK):
    '''
    Convert CMYK to CMY
    C=C*(1-K)+K,
    M=M*(1-K)+K,
    Y=Y*(1-K)+K
    '''

    K=imgCMYK[:,:,3]              # Extract K from CMYK
    K=np.dstack([K,K,K])          # Convert K to 3-D for image operations
    K1=np.ones(K.shape)-K         # Calculate 1-K
    return imgCMYK[:,:,:3]*K1+K   # C=C*(1-K)+K, M=M*(1-K)+K, Y=Y*(1-K)+K

def RGB2CMYK(img):
    '''
    Convert from RGB to CMYK
    '''

    return CMY2CMYK(RGB2CMY(img)) # RGB -> CMY -> CMYK

def CMYK2RGB(img):
    '''
    Convert from CMYK to RGB
    '''

    kimg=np.round(255*CMY2RGB(CMYK2CMY(img))).astype(np.uint8) # CMYK -> CMY -> RGB
    return  kimg

def RGB2HSI(imgRGB):
    '''
    Convert from RGB color code to HSI color code
    I=(R+G+B)/3,               Intensity
    S=1-3*min(R,G,B)/(R+G+B),  Saturation
    H=    t, if B <= G,        Hue
      360-t, otherwise
    where Cos(t)=[(R-G)+(R-B)]/2[(R-G)^2+(R-B)(G-B)]^(1/2)
    where X,Y,Z are the RGB channel intensities of a pixel
    '''

    sm=np.sum(imgRGB,axis=2)                        # calculate sum (r+g+b) for each pixel of the image
    i=sm/3                                          # calculate average intensity (r+g+b)/3 for each pixel of the image
    s=1-np.min(imgRGB,axis=2)/i                     # calculate saturation of each pixel
    b,g,r=imgRGB[:,:,0],imgRGB[:,:,1],imgRGB[:,:,2] # seperate r,g,b components to calculate Hue
    r=r.astype(np.float16)
    g=g.astype(np.float16)
    num=(2*r-g-b)                                   # numerator of Cos(t)
    den=2*np.sqrt(np.square(r-g)+(r-b)*(g-b))       # denominator of Cos(t)
  
    t=np.ones(b.shape,dtype=int)                    # calculate t in degrees
    t=np.divide(num,den,where=den!=0)
    t=np.degrees(np.arccos(t))
    h=np.where(b>g,360-t,t)                         # calculate hue of each pixel

    return np.dstack([h,s,i])                       # create hsi image and return

def HSI2RGB(imgHSI):
    '''
    Convert from HSI color code to RGB color code
    '''
    h,s,i=imgHSI[:,:,0],imgHSI[:,:,1],imgHSI[:,:,2] # extract hue, saturation and intensity components
    b=np.zeros(h.shape,dtype=int)                   # initialise rgb matrices to zero
    g=np.zeros(h.shape,dtype=int)
    r=np.zeros(h.shape,dtype=int)

    t=i*(1-s)
    b=np.where(h<120,t,b)
    r=np.where((120<=h)&(h<240),t,r)
    g=np.where(h>240,t,g)
    
    t=s*(np.cos(np.radians(h))/np.cos(np.radians(60-h))) # for h<120 
    r=np.where(h<120,i*(1+t),r)

    t=s*(np.cos(np.radians(h-120))/np.cos(np.radians(180-h))) # 120<=h<240
    g=np.where((120<=h)&(h<240),i*(1+t),g)

    t=s*(np.cos(np.radians(h-240))/np.cos(np.radians(300-h))) # h>40
    b=np.where(h>240,i*(1+t),b)

    g=np.where(h<120,3*i-r-b,g)
    b=np.where((120<=h)&(h<240),3*i-r-g,b)
    r=np.where(h>240,3*i-g-b,r)

    rgbimg=np.round(np.dstack([b,g,r]))
    rgbimg=rgbimg.astype(np.uint8)
    
    return rgbimg
