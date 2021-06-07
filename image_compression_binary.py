#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from readP5file import readfile
import config
from bitSlicing import slice_bits8,merge_bits_87,merge_bits_876
from matrix_to_binary import creatbinaryeFile

'''
This is the main program that takes a file name from the command line argument, if it is in valid P5 .pgm format, then reads
its pixel values in a matrix and prints it.
'''
if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage <%s> <filename.type>" %sys.argv[0])

    config.initialise()  
    readfile(sys.argv[1])
    print('rows=',config.rows,' columns=',config.columns)
    print(config.A)

    slice_bits8()
#    print('Sliced matrices are created!')
    for i in range(8):
        creatbinaryeFile(sys.argv[1][:-4]+'_b'+str(8-i)+sys.argv[1][-4:],config.bitList[i],config.rows,config.columns,1)

#    print('8 ascii files are created!')
    merge_bits_87()
#    print('merges bit 8 and bit 7')
    creatbinaryeFile(sys.argv[1][:-4]+'_b87'+sys.argv[1][-4:],config.A87,config.rows,config.columns,3)
    merge_bits_876()
    creatbinaryeFile(sys.argv[1][:-4]+'_b876'+sys.argv[1][-4:],config.A876,config.rows,config.columns,7)
