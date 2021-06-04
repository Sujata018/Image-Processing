#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import config 

'''
This function creates 8 different matrices each with the bits in positions
1,2,3,4,5,6,7,8 of the input matrix.
Input  : global variable numpy array A declared in config.py
Outputs: 8 numpy arrays, in a list ,bitList, as declared in config.py -
           bitList[0] contains least significant bit of each elements of A
           bitList[1] contains 2nd bit from right of each elements of A
           bitList[2] contains 3rd bit from right of each elements of A
           bitList[3] contains 4th bit from right of each elements of A
           bitList[4] contains 5th bit from right of each elements of A
           bitList[5] contains 6th bit from right of each elements of A
           bitList[6] contains 7th bit from right of each elements of A
           bitList[7] contains most significant bit of each elements of A

Logic   : Operates on each element, right shifts the byte to get least significant
          bit each time, and copies 0 or 1 to corresponding element in sliced bit arrays
'''
def slice_bits8_old():

    for i in range(8):                          # create list of arrays with all 0s
        config.bitList.append(np.zeros([config.rows,config.columns], dtype=int))
    for i in range(config.rows):
        for j in range(config.columns):         # for each element, slice bits
            gVal=config.A[i][j]                 # store element in gVal
            for bit in range(6):                # check 1st to 6th bits from right
                if gVal%2 == 0:                 # if LSB is 0, put 0 in output matrix position, or else put 1
                    config.bitList[bit][i][j]=0
                else:
                    config.bitList[bit][i][j]=1
                gVal >>= 1                      # bit right-shift to check next bit

            if gVal%2 == 0:                     # check 7th bit
                config.bitList[6][i][j]=0
            else:
                config.bitList[6][i][j]=1

            if gVal>1:                          # check most signofocant bit
                config.bitList[7][i][j]=1
            else:
                config.bitList[7][i][j]=0

'''
This function creates 8 different matrices each with the bits in positions
1,2,3,4,5,6,7,8 of the input matrix.
Input  : global variable numpy array A declared in config.py
Outputs: 8 numpy arrays, in a list ,bitList, as declared in config.py -
           bitList[0] contains least significant bit of each elements of A
           bitList[1] contains 2nd bit from right of each elements of A
           bitList[2] contains 3rd bit from right of each elements of A
           bitList[3] contains 4th bit from right of each elements of A
           bitList[4] contains 5th bit from right of each elements of A
           bitList[5] contains 6th bit from right of each elements of A
           bitList[6] contains 7th bit from right of each elements of A
           bitList[7] contains most significant bit of each elements of A

Logic   : This function uses Pythin's bit unpack in-built function
'''


def slice_bits8():
    config.A=np.array(config.A,dtype=np.uint8)                     # convert matrix to unsigned integer form, so unpackbit function can be applied to it
    unpackA=np.unpackbits(config.A,axis=1)                  # unpack each row of A with 8 bits for each element. E.g. [1,2] -> [0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0]
    for bit in range(8):                             # create bit sliced matrices
        bitIndex=np.arange(bit,(config.columns)*8,8) # index list for index slicing. E.g. to slice all 0th bit index ->[0,8,16,24,...]
        config.bitList.append(unpackA[:,bitIndex])   # store bit-sliced matrices in a global list

'''
Merges the most signoficant bit and next significant bit to produce a visible image
'''
def merge_bits_87():
    config.A87 = (2* config.bitList[0])+config.bitList[1]
    
'''
Merges three most signoficant bits to produce a visible image
'''
def merge_bits_876():
    config.A876 = 4*config.bitList[0]+ 2* config.bitList[1]+config.bitList[2]
