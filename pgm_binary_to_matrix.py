#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from readP5file import readfile


'''
This is the main program that takes a file name from the command line argument, if it is in valid P5 .pgm format, then reads
its pixel values in a matrix and prints it.
'''
if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage <%s> <filename.type>" %sys.argv[0])

    A,rows,columns,maxGray=readfile(sys.argv[1])
    print('rows=',rows,' columns=',columns)
    print(A)

