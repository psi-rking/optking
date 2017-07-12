from math import fabs
import numpy as np

### Linear algebra routines.
def absMax(V):
    return max(abs(elem) for elem in V)

def absMin(V):
    return min(abs(elem) for elem in V)

def rms(V):
    return np.sqrt(np.mean(V**2))

def signOfDouble(d):
    if d>0: return 1
    elif d<0: return -1
    else: return 0

# Returns eigenvectors as rows?
def symmMatEig(mat):
    evals, evects = np.linalg.eigh(mat)
    evects = evects.T
    return evals, evects

import operator
# returns eigenvectors as rows; orders evals
def asymmMatEig(mat):
    evals, evects = np.linalg.eig(mat)
    evects = evects.T
    evalsSorted, evectsSorted = zip(*sorted(zip(evals,evects),key=operator.itemgetter(0)))
    # convert from tuple to array
    evalsSorted = np.array( evalsSorted, float)
    evectsSorted = np.array( evectsSorted, float)
    return evalsSorted, evectsSorted

#  Return the inverse of a real, symmetric matrix.  If "redundant" == true,
#  then a generalized inverse is permitted.
def symmMatInv(A, redundant=False, redundant_eval_tol=1.0e-10):
    dim = A.shape[0]
    if dim <= 0: return np.zeros( (0,0), float)
    det = 1.0

    try:
        evals, evects = symmMatEig(A)
    except:
        raise INTCO_EXCEPT("symmMatrixInv could not diagonalize")

    for i in range(dim):
        det *= evals[i]

    if not redundant and fabs(det) < 1E-10:
        raise INTCO_EXCEPT("symmMatrixInv non-generalized inverse of matrix failed")

    diagInv = np.zeros( (dim,dim), float)

    if redundant:
        for i in range(dim):
            if fabs(evals[i]) > redundant_eval_tol:
                diagInv[i,i] = 1.0/evals[i]
    else:
        for i in range(dim):
            diagInv[i,i] = 1.0/evals[i]

    # A^-1 = P^t D^-1 P
    tmpMat = np.dot(diagInv, evects)
    AInv = np.dot(evects.T, tmpMat)
    return AInv

def symmMatRoot(A, Inverse = False):
    evals, evects = np.linalg.eigh(A)
    rootMatrix = np.zeros((len(evals), len(evals)), float)
    if (Inverse):
        for i in range (0,len(evals)):
            evals[i] = 1 / evals[i]
    
    Q = np.zeros((len(evals), len(evals)), float)
    for i in range (len(evals)):
        for j in range (len(evects)):
            Q[j][i] = evects[j]
                
    for i in range (0, len(evals)):
        rootMatrix[i][i] = sqrt(evals[i]) 
    
    A = np.dot(Q, np.dot(rootMatrix, Q.T))        
    
    return A        

