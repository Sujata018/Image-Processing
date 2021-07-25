import sys
import numpy as np
from sklearn import neighbors

'''
This package contains techniques for K - Nearest Neighbourhood classification
of categorical data
'''

def knn(X,y,img,n_neighbors=9):
    '''
    K-Nearest-Neighbors classifier.
    Default k=9
    Inputs: X (training data), n x d, n -> # of samples, d -> dimension
            y (target data), binary, n x 1
            n_neighbors (K)
    '''
    clf=neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X,y)                             # fit sample and target to create a KNN model

    imgrow,imgcol,imgdepth=img.shape
    simg=img
    simg=img.reshape(imgrow*imgcol,imgdepth) # stack input image
    
    z=clf.predict(simg)                      # predict class for each image pixe; using the model
 
    for i in range(simg.shape[0]):           
        if z[i]==0:                          # class 0 pixels are colored in blue
            simg[i]=[255,0,0]
        elif z[i]==1:
            simg[i]=[0,255,0]                # class 1 pixels are colored in green
    simg=simg.reshape(imgrow,imgcol,imgdepth)# destack classified image to its original shape 
    
    return simg                              # return classified image

