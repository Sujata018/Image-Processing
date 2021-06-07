''' This file contains global variables used in image processing package
'''
import numpy as np

def initialise():
    global A        # image pixel intensity matrix
    global rows     # number of rows in image intensity matrix
    global columns  # number of columns in image intensity matrix
    global maxGray  # maximum intensity possible in image = 2**N -1, N: number of bits used
    global hist     # 1-D histogram of the intensity matrix
    global bitList  # list of matrices of bits after bit slicing 
    global A87      # image intensity matrix after comibing bits 7 and 8
    global A876
    

    A=[]
    rows=0
    columns=0
    maxGray=0
    hist=[]
    bitList=[]
    A87=[]
    A876=[]
