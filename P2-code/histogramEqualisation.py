import sys
import cv2 as cv
import numpy as np


'''
To normalize different lighting conditions, perform histogram equalization
on each chanel of the images
'''

#stemfile="lung.jpg"
for i in range(1,2):
#    file='national-cancer-institute.jpg'
#    file='43601.jpg'
#    file=stemfile[:-4]+str(i)+stemfile[-4:]
    file='lung.jpg'
    img=cv.imread(file)                    # convert file to image tensor

    if img is None:
        sys.exit("Could not read the image "+file)

    b,g,r=img[:,:,0],img[:,:,1],img[:,:,2] # split into blue, green, red
    cv.imwrite(file[:-4]+'_b.bmp',b)
    cv.imwrite(file[:-4]+'_g.bmp',g)
    cv.imwrite(file[:-4]+'_r.bmp',r)

    bequ=cv.equalizeHist(b)                # equalize histogram of blue chanel 
    gequ=cv.equalizeHist(g)                # green chanel
    requ=cv.equalizeHist(r)                # red chanel

    imgequ=np.dstack([bequ,gequ,requ])     # join histogram equalized chanels to image tensor

    cv.imwrite(file[:-4]+'_HE.bmp',imgequ) # output image name extended to _HE
    cv.imwrite(file[:-4]+'_HE_b.bmp',bequ)
    cv.imwrite(file[:-4]+'_HE_r.bmp',requ)
    cv.imwrite(file[:-4]+'_HE_g.bmp',gequ)
