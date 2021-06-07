#!/usr/bin/env python
# coding: utf-8

import sys                     # for system exit in case of errors
import numpy as np             # to store the matrix in numpy array


'''
Create a portable gray map P5 (binary) format file from a matrix
Inputs: A        : the pixel matrix
        rows     : number of rows in A
        columns  : number of columns in A
        maxGray  : maximum gray bit range
        filename : name of the output file
Output: the .pgm file with the filename, the matrix pixels encoded in binary in it. 
'''
def createbinaryFile(filename,A,rows,columns,maxGray):
    try:
        with open(filename,'wb') as f:                                  # open the file in binary write mode
            f.write(bytearray("P5\n",'utf-8'))                          # encoded magic number
            f.write(bytearray(str(columns)+' '+str(rows)+'\n', 'utf-8'))# encoded column and row information, separated by whitespace
            f.write(bytearray(str(maxGray)+'\n','utf-8'))               # encoded maxgray information
            for row in range(rows):
                f.write(bytearray(list(A[row])))                        # encoded each row of the matrix in a single line
    except:                                       
        sys.exit("Error in writing to file "+sys.argv[1])               # exit if any error
