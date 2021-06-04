#!/usr/bin/env python
# coding: utf-8

import sys                     # for system exit in case of errors
import numpy as np             # to store the matrix in numpy array


'''
Create a portable gray map P2 (ASCII) format file with name passed from calling program,
and the matrix information passed from the calling program
'''
#def createASCIIFile(A,rows,columns,maxGray,filename):   
#    try:
#        with open(filename,'w') as f:            # open the file in write mode
#            print("P2",file=f)                   # print magic number
#            print(columns," ",rows,file=f)       # print column and row information, separated by whitespace
#            print(maxGray,file=f)                # print maxgray information
#            for row in range(rows):
#                print(*A[row],file=f)            # print the matrix in row major order seperated by white spaces, each row in a single line
#    except:                                      # exit if any error 
#        sys.exit("Error in writing to file"+sys.argv[1])

'''
   Used numpy function savetxt to reduce time for writing a big matrix to file.
   parameters passed : filename      <- name of the output file
                       A             <- the numpy array to be written in output
                       fmt='%d'      <- writing each element of A in integer format
                       header= '...' <- header information to be written before the array integers (contains header for the pgm file)
                       comments=''   <- by specifying null string as comment, I am making sure the heaser above is not commented. 
                                        If this parameter is not specified, the header information lines will start with '#', as default.
'''
def createASCIIFile(A,rows,columns,maxGray,filename): 
    np.savetxt(filename,A,fmt='%d',header='P2\n'+ str(columns) + ' ' + str(rows) + '\n' + str(maxGray),comments='')
