import numpy as np
from numpy.linalg import multi_dot
import cv2 as cv

'''
This package contains techniques related to MDC (most discriminant color) space
'''

def RGB2MDC(img,A=1,file=0):
    '''
    Transform input image from RGB space to MDC (most discrimination color) space,
    by matrix multiplication MDCimage = RGBimage * A
   
    Input : img  -> RGB image of dimension m x n x 3
            A    -> transformation matrix of size 3 x 3. Default is identity matrix
            file -> name of the file (with extensions) to save the output image
                    if file is not provided, then output image is not saved
    Output: transformed image of dimensions same as input image
    '''
    if type(A) == int:
        A=np.eye(3)
    mimg=multi_dot([img,A])
    if file !=0:
        cv.imwrite(file,mimg)
        c,d,m=mimg[:,:,0],img[:,:,1],img[:,:,2]  # split into m, d, c channels
        cv.imwrite(file[:-4]+'_m.bmp',m)
        cv.imwrite(file[:-4]+'_d.bmp',d)
        cv.imwrite(file[:-4]+'_c.bmp',c)

    return mimg
