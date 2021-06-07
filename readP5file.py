import numpy as np
import sys
import config

'''
Function to read next uncommented line from the input file 
'''
def readnextUncommentedLine(f):
    comment = True           # Set parameter that identifies if the input line is a commnet line

    while (comment == True): # keep reading until an uncommented line is found
        line=f.readline()    # read next line
        if line[0] != '#':   # if first character of the line is #, then it is a commented line
            comment = False
    return line              # return the first uncommneted line

'''
This function reads the inout pgm file of P5 format and stores the pixel data in a matrix A, passes it back to the calling program
'''
def readfile(filename):
    m=[]                     # list to store the matrix
    headers=[]               # list to store header informantion
    print("columns=",config.columns,"rows=",config.rows,"maxGray=",config.maxGray)
    try:
        f=open(filename,'rb')# read file in binary mode  
    except:
        sys.exit("invalid PGM format")

    while len(headers)<4:    # obtain the magic number, column, row and max gray header informations from same of different lines
        header = readnextUncommentedLine(f) # keep reading uncommneted lines until all 4 header informations are obtained
        headers.extend(header.split())

    magicNumber = headers[0].decode('ascii')
    config.columns,config.rows,config.maxGray=list(map(int,headers[1:]))
    print("magicNumber=",magicNumber,"columns=",config.columns,"rows=",config.rows,"maxGray=",config.maxGray)
    
    if magicNumber=='P5':
        print("it's P5 file!")
        for line in f:
            for b in line:
                m.append(b)  # read all pixel values in a list
        f.close()
        config.A=np.array(m).reshape(config.rows,config.columns) # create numpy array
            
    else:
        sys.exit("invalid PGM file")

  
