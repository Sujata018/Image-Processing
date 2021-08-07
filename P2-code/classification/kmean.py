import numpy as np
from sklearn.cluster import KMeans

'''
This package contains techniques for K - Means classification
'''

def kmeans(X,K=20):
    '''
    K-Means clustering.
    Default k=20
    Inputs: X (training data), n x d, n -> # of samples, d -> dimension
            K -> number of desired clusters
    Output: cluster_centers_  -> The 20 cluster centerd identified by K-NN
    '''
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    
    return kmeans.cluster_centers_     # return cluster centers
