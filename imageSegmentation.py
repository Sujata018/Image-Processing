import sys

import config
from readP5file import readfile
from histogram import  createHistogram,plotHistogram
from threshold import thresholdHalf,thresholdMeanOfMeans,thresholdOtsu
from matrix_to_binary import creatBinaryeFile

if __name__=='__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage <%s> <input_filename.type> <output_filename.type>" %sys.argv[0])

    config.initialise()        # initialise global variables

    readfile(sys.argv[1])      # store input image intensities to global matrix A

    createHistogram()          # create 1-D histogram from image
    plotHistogram(config.hist,sys.argv[1]) # save the histogram

    SA=thresholdHalf()         # segment image with threshold = L/2
    creatBinaryeFile(sys.argv[1][:-4]+'_segmented half'+sys.argv[1][-4:],SA,config.rows,config.columns,config.maxGray)

    SA=thresholdMeanOfMeans()  # segment image with mean of mean threshold
    creatBinaryeFile(sys.argv[1][:-4]+'_segmented mom'+sys.argv[1][-4:],SA,config.rows,config.columns,config.maxGray)
    
    SA=thresholdOtsu()        # segment image with Otsu's thresholding method
    creatBinaryeFile(sys.argv[1][:-4]+'_segmented Otsu'+sys.argv[1][-4:],SA,config.rows,config.columns,config.maxGray)
    
