#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import config
from matrix_to_ascii import createASCIIFile
from readP5file import readfile

if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage <%s> <filename.type>" %sys.argv[0])

    config.initialise()  
    readfile(sys.argv[1])
    createASCIIFile(config.A,config.rows,config.columns,config.maxGray,sys.argv[1][:-4]+'_ASCII'+sys.argv[1][-4:])    
