#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import config

def readfile(filename):
    m=[]

    try:
        with open(filename) as f:
            lines=f.readlines()
            for l in lines:
                if l[0] == '#':
                    lines.remove(l)
    except:
        sys.exit("invalid PGM format")

    magicNumber = lines[0].split()[0]
    if magicNumber=='P2':
        print("it's an ascii file!")
        for line in lines[1:]:
            try:
                m.extend(list(map(int,line.split())))
            except:
                sys.exit("invalid PGM format")
        config.columns,config.rows,config.maxGray=m[0:3]
        config.A=np.array(m[3:]).reshape(config.rows,config.columns)     
       
    else:
        sys.exit("invalid PGM file")
