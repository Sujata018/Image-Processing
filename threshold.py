import sys
import config
import numpy as np
from statistics import mean,stdev
from histogram import normalise,plotHistogram

'''
Threshold an image based on a single threshold
'''
def applySingleThreshold(T):

    SA=np.zeros((config.rows,config.columns),dtype=int)
    for i in range(config.rows):
        for j in range(config.columns):
            if config.A[i,j]>T:         # pixel is white if intensity > L/2
                SA[i,j]=255
            else:
                SA[i,j]=0         # pixel is black if intensity <= L/2
    return SA

'''
Threshold an image based on a L/2
'''

def thresholdHalf():
    T= config.maxGray/2           # define threshold = L/2
    return (applySingleThreshold(T))   # return matrix after applying threshold L/2

'''
Calculate expectation of a given 1-D array
Inputs: X : 1-D array name
        normal : =1 if the histogram is normalised, 0 otherwise. Default : 0.
        optional starting index : default 0. If noy passed, end index also not to be passed.
        optional end index : default index of the last element in the array
    
Output: Expectation = sum(value of index * element)/sum(values of indices)
Logic: To calculate expected value of a histogram, where each frequency is stored at index same as its label,
       E(A)=sum(x.p(x))/sum(p(x))
       For a normalised histogram, sum(p(x))=1
'''
def expectation(X,normal=0,start=0,end=0):
    if end == 0:      # if end position is not provided, calculate default as last position
        end = len(X)-1

    SumxP = 0         # initialise variable for sum of x*P(x)
    if normal == 1:   # if normalised histogram, no need to calculate sum of frequencies, as sum p(x)=1
        SumP = 1
    else:
        SumP = 0
        
    for i in range(start,end+1):
        
        try:
            SumxP += i*X[i]
        except:
            sys.exit('out of bound i '+str(i))

        if normal == 0:
            SumP += X[i]

    if SumP == 0:
        return 0
    else:
        return SumxP/SumP


'''
Threshold an image based on a mean of mean.
Starting threshold T= L/2, calculate means m1, m2 of intensities lower and higher then T.
new T= (m1+m2)/2
Repeat recalculating until dT < a given number
'''

def thresholdMeanOfMeans(Trange=1):
    T= config.maxGray                  # initialise threshold = L/2
    T >>= 1    
    Told=0                             # threshold obtained from previous iteration. Initialised to 0.

    while (abs(T-Told)>Trange):        # iterate until difference between new and old threshold is less than provided Trange      
                      
        m1 = expectation(config.hist,0,0,int(T)) # calculate expectation of config.hist[0:T+1],0-> histogram not normalised,0->start index,T->end index
        m2 = expectation(config.hist,0,int(T)+1) # calculate expectation of config.hist[T+1:],0-> histogram not normalised,T+1->start index

        Told=T                         # threshold obtained from previous iteration. 
        T=(m1+m2)/2                    # new threshold = mean of m1 and m2

    print("final threshold (mean of means) = ",T)
    return (applySingleThreshold(T))   # return matrix after applying threshold 


'''
Decide threshold based on Otsu's method
'''

def thresholdOtsu():

    H=normalise(config.hist)

    m = expectation(H)                 # Calculate mean intensity of the whole image
    p = H[0]                           # stores sum(p(x)) for current hypothesis threshold i in the loop
    m1 = 0                             # stores sum(xp(x)) for current hypothesis threshold i in the loop
    s=0                                # inter-class variance for the for current hypothesis threshold i in the loop
    splot=[]                           # list of inter-class variances for plotting
    
    for i in range(1,len(H)):          # For each possible threshold, calculate inter class variance      
                      
        p += H[i]                      # include p(i) in the running sum 
        m1 += i*H[i]                   # include i*p(i) in the running sum

        if p!=0 and p!= 1:
            s=(m1-m*p)**2/(p*(1-p))
        splot.append(s)

    plotHistogram(splot,'variance.png','inter-class variance') # plot interclass variance against intensity
    
    k = splot.index(max(splot))         # The intensity where inter class variance is maximum, is the threshold  
    print("final threshold (Otsu)= ",k)
    return (applySingleThreshold(k))    # return matrix after applying threshold 

