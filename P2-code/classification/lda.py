import sys
import numpy as np
from scipy.linalg import eigh,eig
from numpy.linalg import multi_dot

'''
This package contains techniques for Linear Discremination Analysis for classification of categorical data
'''

# Global variables

N=0      # Total number of test samples        
Psb=[]   # Matrix representation of between-class spreads for optimising P
PsbT=[]  # Block transpose of Psb above
Psw=[]   # Matrix representation of within-class spreads for optimising P
PswT=[]  # Block transpose of Psw above
Ysb=[]   # Matrix representation of between-class spreads for optimising A
YsbT=[]  # Block transpose of Ysb above
Ysw=[]   # Matrix representation of within-class spreads for optimising A
YswT=[]  # Block transpose of Ysw above
A=[]     # Transformation matrix for rgb to mdc
P=[]     # Projection matrix for Fisher-Rao optimization of LFT features
Qp=[]    # Positive samples
Qn=[]    # Negative samples

def calcScatterWithA():
    '''
    Calculate inter and intra scatter matrices for a given color space obtained from RGB by transformation matrix A

    Sb = Psb,Asb,np.transpose(Asb),PsbTranspose /2
    Sw = Psw,Asw,np.transpose(Asw),PswTranspose /N

    2: # of classes
    N: # of samples
    '''
    global N,Psb,PsbT,Psw,PswT, A

    Z=np.zeros([3,3])                                                 #      | A  0 |
    Asb=np.block([[A,Z],[Z,A]])    # create Asb as block diagonal with A ->  | 0  A |

    B=[]                           # create Asw as block diagonal with A ->  | A 0 ... 0 | T                          
    for i in range(N):             #                                         | 0 A ... 0 | N
        R=[Z for _ in range(N)]    #                                          ...........  |
        R[i]=A                     #                                         | 0 0 ... A | -
        B.append(R)                #                                         <----N------>

    Asw=np.block(B)

    Sb=multi_dot([Psb,Asb,np.transpose(Asb),PsbT])/2 # Between class scatter
    Sw=multi_dot([Psw,Asw,np.transpose(Asw),PswT])/N # Within class scatter

    return Sb,Sw

def optimizeP():
    '''
    Find optimum P (projection matrix) for a fixed A (coefficient matrix for RGB -> MDC transform)
    by solving the generalised Eigenvalue problem  C.P=B.P.Al
        where C=Psb.Asb.Transpose(Psb.Asb)/c,
              B=Psw.Asw.Transpose(Psw.Asw)/N,
              P=projecttion matrix, which is the matix with columns same as the eigen vectors
              Al=the diagonal matrix with the eigen values as its diagonal elements
    '''
    global P

    Sb,Sw=calcScatterWithA()

    w,P=eigh(Sb,Sw)               # solve the generalised eigen value problem to get P (eigen verctors)

    return np.fliplr(P[:,-3:])    # Last 3 cColumns of P are the eigenvectors correspond to the largest 3 eigen values 
 
def optimizeA():
    '''
    Find optimum A (coefficient matrix for RGB -> MDC transform) for a fixed P (projection matrix)
    by solving the generalised Eigenvalue problem  C.AP=B.A.Al
        where C=Transpose(Psb).Asb.Transpose(Asb).Psb/c,
              B=Transpose(Psw).Asw.Transpose(Asw).Psw/N,
              P=projecttion matrix, which is the matix with columns same as the eigen vectors
              Al=the diagonal matrix with the eigen values as its diagonal elements
    '''
    global N,Ysb,YsbT,Ysw,YswT, A, P

    Z=np.zeros([8,3])              #                                         | P  0 |
    Asb=np.block([[P,Z],[Z,P]])    # create Asb as block diagonal with P ->  | 0  P |

    B=[]                           # create Asw as block diagonal with A ->  | P 0 ... 0 | T                          
    for i in range(N):             #                                         | 0 P ... 0 | N
        R=[Z for _ in range(N)]    #                                          ...........  |
        R[i]=P                     #                                         | 0 0 ... P | -
        B.append(R)                #                                         <----N------>

    Asw=np.block(B)

    
    # Left hand side of generalised eigenvalue problem (C)
    ls=multi_dot([YsbT,Asb,np.transpose(Asb),Ysb])/2
    # Right hand side of generalised eigenvalue problem (B)
    rs=multi_dot([YswT,Asw,np.transpose(Asw),Ysw])/N

    w,A=eigh(ls,rs)                # solve the generalised eigen value problem to get A (eigen verctors)

    return np.fliplr(A)  

def calculateJ():
    '''
    Calculate J, the ratio of inter class scatter to the intra-class scatter,
    which is maximised in Fisher Rao linear discrimination model for classification.

       N.det(transpose(P)sum((Mi-M)A.transpose((Mi-M).A)).P)
    J= -------------------------------------------------------,
       c.det(transpose(P)sum((Qij-Mi)A.transpose((Qij-Mi).A))P)
       c=number of classes. c=2 in this code.
 
    '''
    global A,P
    
    # calculate inter-class scatter (numerator of J) and intra-class scatter (denominator of J)
    Sb,Sw=calcScatterWithA()
    num=np.linalg.det(multi_dot([np.transpose(P),Sb,P]))
    den=np.linalg.det(multi_dot([np.transpose(P),Sw,P]))

    # calculate J and return
    return num/den

def initialise(Qpos,Qneg):
    '''
    Initialise matrices used for fisher-rao optimization
    '''

    global N,Psb,PsbT,Psw,PswT,Ysb,YsbT,Ysw,YswT,A,P,M,M1,M2,Qp,Qn

    Qp,Qn=Qpos,Qneg                      # store input sample data to global variable 
    n1,n2=Qp.shape[0],Qn.shape[0]        # number of positive and negative samples
    N=n1+n2                              # total number of sampples

    M=np.mean(np.vstack([Qp,Qn]),axis=0) # calculate sample mean
    M1=np.mean(Qp,axis=0)                # mean of positive class
    M2=np.mean(Qn,axis=0)                # mean of negative class
    
    M=M.reshape((8,3),order='F')         # convert means from vector to 8(lft) * 3(rgb) matrix
    M1=M1.reshape((8,3),order='F')       
    M2=M2.reshape((8,3),order='F')       
 
    # Creating scatter matrices for the matrix representation of the Fisher Rao optimization forrmula
    Psb=np.hstack([M1-M,M2-M])           # between class scatters for optimising P    # Populate matrices used to optimise P
    PsbT=np.vstack([np.transpose(M1-M),np.transpose(M2-M)])                           # Psb  = [M1-M       M2-M]                 : 8 x 3c
    Ysb=np.vstack([M1-M,M2-M])           # between class scatter for optimising A     # PsbT = [transpose(M1-M)]                 : 3c x 8
    YsbT=np.hstack([np.transpose(M1-M),np.transpose(M2-M)])                           #        [transpose(M2-M)]
                                                                                      # Psw  = [Qij-M1...  Qij-M2 ...]           : 8 x 3N
    for i in range(Qp.shape[0]):         # calculate Qij-Mi for positive samples      # PswT = [transpose(Qij-M1)]               : 3N x 8 
        q=Qp[i]                                                                       #          ...   
        q=q.reshape((8,3),order='F')                                                  #        [transpose(Qij-M2)]
        if i ==0:                                                                     #          ...   
            Psw=Ysw=q-M1                                                              # Ppulate matrices used to optimise A
            PswT=YswT=np.transpose(q-M1)                                              # Ysb =  [M1-M]                            : 8c x 3
        else:                                                                         #        [M2-M] 
            Psw=np.hstack([Psw,q-M1])                                                 # Ysbt = [transpose(M1-M) transpose(M2-M)] : 3 x 8c
            PswT=np.vstack([PswT,np.transpose(q-M1)])                                 # Ysw =  [Qij-M1]                          : 8N x 3
            Ysw=np.vstack([Ysw,q-M1])                                                 #          ...
            YswT=np.hstack([YswT,np.transpose(q-M1)])                                 #         Qij-M2
                                                                                      #          ...  ]
    for q in Qn:                         # calculate Qij-M1 for negative samples      # YswT = [transpose(Qij-M1)... transpose(Qij-M2)...] : 3 x 8N
        q=q.reshape((8,3),order='F')
        Psw=np.hstack([Psw,q-M2])
        PswT=np.vstack([PswT,np.transpose(q-M2)])
        Ysw=np.vstack([Ysw,q-M2])                                                 
        YswT=np.hstack([YswT,np.transpose(q-M2)])                                 
      
    A=np.eye(3)                          # initialise A to identity matrix
   

def fisherRaoOptimization(Qpos,Qneg,deltaJ):
    '''
    Iterative Fisher-Rao Optimization

    This function is an iterative process to optimize
    A: the coefficient matrix to transform from RGB color space to MDC (most discriminant color) space
    P: the projection matrix for LFT features extracted from the MDC color space to optimally
       separate out nuclie and extra cellular classes.

    by solving following optimization problem:
    
    {A*,P*}=arg (A,P) max (Pt.Sb.P)/(Pt.Sw.P)

    The detailed algorithm is described in section E. of paper:
   
    Partitioning Histopathological Images: An Integrated Framework for Supervised
    Color-Texture Segmentation and Cell Splitting.

    by Hui Kong*, Metin Gurcan, Senior Member, IEEE, and Kamel Belkacem-Boussaid, Senior Member, IEEE

    Inputs:
        Qpos   : n x 24 matrix of n positive samples of dimension 24 
        Qneg   : n x 24 matrix of n negative samples of dimension 24
        deltaJ : convergence criterion for iterative calculation of J (the ratio of between class spread / within class spread)
    '''
    global P

    initialise(Qpos,Qneg)            # initialise matrices for optimization
    P=optimizeP()                    # find optimal projection matrix P for the current rgb to mdt map matrix A using Fisher Rao optimization
    J=calculateJ()                   # calculate ratio of between scatter to within scatter
    print("Initial J=",J)
    niter=100                        # maximum number of iterations
    Jnew=0                           # stores recalculated J after every iteration

    pltJ=[J]                         # 1-D array to store the J values in each iteraton
    for i in range(niter):
        A=optimizeA()                # maximize A keeping P fixed
        P=optimizeP()                # maximise P keeping A fixed
        Jnew=calculateJ()            # calculate J with current A and P
        pltJ.append(Jnew)            # add J for current iteration to 1-D array for plot
        print("iteration ", i+1,"J=",Jnew)
        if abs(Jnew-J)<=deltaJ:      # stop iterating if not much change in J
            print("Converged at J=",Jnew)
            break
        J=Jnew
    
    return A,P,pltJ                  # return optimised A, P and the plotting data for iterations of J

