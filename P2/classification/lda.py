import sys
import numpy as np
from scipy.linalg import eigh,eig
from numpy.linalg import multi_dot

'''
This package contains techniques for Linear Discremination Analysy
for classification of categorical data
'''
N=0
Psb=[]
Psw=[]
Ysb=[]
Ysw=[]
A=[]
P=[]
Qp=[]
Qn=[]

def optimizeP():
    '''
    Find optimum P (projection matrix) for a fixed A (coefficient matrix for RGB -> MDC transform)
    by solving the generalised Eigenvalue problem  C.P=B.P.Al
        where C=Psb.Asb.Transpose(Psb.Asb)/c,
              B=Psw.Asw.Transpose(Psw.Asw)/N,
              P=projecttion matrix, which is the matix with columns same as the eigen vectors
              Al=the diagonal matrix with the eigen values as its diagonal elements
    '''
    global N,Psb,Psw, A, P

    Z=np.zeros([3,3])                                                 #      | A  0 |
    Asb=np.block([[A,Z],[Z,A]])    # create Asb as block diagonal with A ->  | 0  A |

    B=[]                           # create Asw as block diagonal with A ->  | A 0 ... 0 | T                          
    for i in range(N):             #                                         | 0 A ... 0 | N
        R=[Z for _ in range(N)]    #                                          ...........  |
        R[i]=A                     #                                         | 0 0 ... A | -
        B.append(R)                #                                         <----N------>

    Asw=np.block(B)

    ls=multi_dot([Psb,Asb,np.transpose(Asb),np.transpose(Psb)])/2 # Left hand side of generalised eigenvalue problem (C)
    rs=multi_dot([Psw,Asw,np.transpose(Asw),np.transpose(Psw)])/N # Right hand side of generalised eigenvalue problem (B)

    w,P=eigh(ls,rs)               # solve the generalised eigen value problem to get P (eigen verctors)

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
    global N,Ysb,Ysw, A, P

    Z=np.zeros([8,3])              #                                         | P  0 |
    Asb=np.block([[P,Z],[Z,P]])    # create Asb as block diagonal with P ->  | 0  P |

    B=[]                           # create Asw as block diagonal with A ->  | P 0 ... 0 | T                          
    for i in range(N):             #                                         | 0 P ... 0 | N
        R=[Z for _ in range(N)]    #                                          ...........  |
        R[i]=P                     #                                         | 0 0 ... P | -
        B.append(R)                #                                         <----N------>

    Asw=np.block(B)

    
    # Left hand side of generalised eigenvalue problem (C)
    ls=multi_dot([np.transpose(Ysb),Asb,np.transpose(Asb),Ysb])/2
    # Right hand side of generalised eigenvalue problem (B)
    rs=multi_dot([np.transpose(Ysw),Asw,np.transpose(Asw),Ysw])/N

    w,A=eigh(ls,rs)                # solve the generalised eigen value problem to get A (eigen verctors)

    return np.fliplr(A)  

def calculateJ(A,P):
    '''
    Calculate J, the ratio of inter class scatter to the intra-class scatter,
    which is maximised in Fisher Rao linear discrimination model for classification.

       N.tr(transpose(P)sum((Mi-M)A.transpose((Mi-M).A)).P)
    J= -------------------------------------------------------,
       c.tr(transpose(P)sum((Qij-Mi)A.transpose((Qij-Mi).A))P)
       c=number of classes. c=2 in this code.
 
    '''
    global Psw, Psb,N
    
    # calculate inter-class scatter (numerator of J)
    At=np.transpose(A)
    Sb=0
    for i in range(0,6,3):
        MM=Psb[:,i:i+3]
        Sb += multi_dot([MM,A,At,np.transpose(MM)])    
    Sb=multi_dot([np.transpose(P),Sb,P])
    num=N*np.trace(Sb)

    # calculate intra-class scatter (denominator of J)
    sumQA=0
    for i in range(0,3*N,3):
        QM=Psw[:,i:i+3]
        sumQA += multi_dot([QM,A,np.transpose(A),np.transpose(QM)])
    Sw=multi_dot([np.transpose(P),sumQA,P])
    den=2*np.trace(Sw)

    # calculate J and return
    return num/den

def initialise(Qpos,Qneg):
    '''
    Initialise matrices used for fisher-rao optimization
    '''

    global N,Psb,Psw,Ysb,Ysw,A,P,M,M1,M2,Qp,Qn

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
    Psb=np.hstack([M1-M,M2-M])           # between class scatter for optimising P
    Ysb=np.vstack([M1-M,M2-M])           # between class scatter for optimising A

    for i in range(Qp.shape[0]):         # calculate Qij-Mi for positive samples
        q=Qp[i]
        q=q.reshape((8,3),order='F')
        if i ==0:
            Psw=Ysw=q-M1
        else:
            Psw=np.hstack([Psw,q-M1])
            Ysw=np.vstack([Ysw,q-M1])
            
    for q in Qn:                         # calculate Qij-M1 for negative samples
        q=q.reshape((8,3),order='F')
        Psw=np.hstack([Psw,q-M2])
        Ysw=np.vstack([Ysw,q-M2])
      
    A=np.eye(3)                          # initialise A to identity matrix
   

def fisherRaoOptimization(Qpos,Qneg):
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
        Qpos : n x 24 matrix of n positive samples of dimension 24 
        Qneg : n x 24 matrix of n negative samples of dimension 24
    '''

    global N,Psb,Psw, A, P,M,M1,M2, Qp, Qn

    initialise(Qpos,Qneg)       # initialise matrices for optimization

    P=optimizeP()               # find optimal projection matrix P for the current rgb to mdt map matrix A using Fisher Rao optimization
    print("P=",P)
    J=calculateJ(A,P)              # calculate ratio of between scatter to within scatter
    
    niter=30                    # maximum number of iterations
    Jnew=0                      # stores recalculated J after every iteration
    deltaJ=0.001                # convergence criterion for J

    for i in range(niter):
        A=optimizeA()
        P=optimizeP()
        Jnew=calculateJ(A,P)
 #       print("iteration ", i+1,"J=",J, " Jnew = ",Jnew,"Jnew-J=",Jnew-J)
        if abs(Jnew-J)<=deltaJ:
  #          print("Converged at J=",Jnew)
            break
        J=Jnew
    return A,P
