import sys
import numpy as np
from sklearn import neighbors

'''
This package contains techniques for K - Nearest Neighbourhood classification
'''

def knn(X,y,img,n_neighbors=9):
    '''
    K-Nearest-Neighbors classifier.
    Default k=9
    Inputs: X (training data), n x d, n -> # of samples, d -> dimension
            y (target data), binary, n x 1 
            img (image)
            n_neighbors (K)
    Output: simg (pixel in blue for class 0, pixel in color green for class 1)
    '''
    clf=neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X,y)                               # fit sample and target to create a KNN model

    imgrow,imgcol,imgdepth=img.shape
    iimgrow=imgrow*imgcol
    iimg=img.reshape(iimgrow,imgdepth)         # stack input image
    simgdepth=3                                # create 3D image with last dimension as 3 for color coding prediction results
    simg=np.zeros([iimgrow,simgdepth],dtype=int) 
    
    z=clf.predict(iimg)                        # predict class for each image pixe; using the model
 
    for i in range(iimgrow):           
        if z[i]==0:                            # class 0 pixels are colored in blue
            simg[i]=[255,0,0]
        elif z[i]==1:
            simg[i]=[0,255,0]                  # class 1 pixels are colored in green
    simg=simg.reshape(imgrow,imgcol,simgdepth) # destack classified image to its original shape 
    
    return simg                                # return classified image

