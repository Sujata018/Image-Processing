import cv2 as cv
import sys
import os


if __name__=='__main__':

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print('File list obtained:',files)
    os.mkdir('JPG')
    for file in files:
        filename,xtn=file.split('.')
        if xtn != 'py':
            img=cv.imread(file)

            if img is None:
                sys.exit("Could not read the image.")

            print("size of ",file," : ",img.shape)
            cv.imwrite('JPG/'+filename+'.jpg',img)
        
