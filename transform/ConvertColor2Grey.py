import cv2 as cv
import os

if __name__=='__main__':
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for filename in files:
        img=cv.imread(filename)
        if img is not None:
            oimg=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imwrite(filename[:-4]+'_Grey'+filename[-4:],oimg)
