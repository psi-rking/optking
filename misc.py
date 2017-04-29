from math import fabs
import numpy as np

### Linear algebra routines.

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


def isDqSymmetric(intcos, geom, Dq):
    print '\tTODO add isDqSymmetric'
    return True

def symmetrizeXYZ(XYZ):
    print '\tTODO add symmetrize XYZ'
    return XYZ

### Simple array operations.

def delta(x, y):
    if x == y: return 1
    return 0

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

### Printing functions.

def printMat(M):
    for row in range(M.shape[0]):
       for col in range(M.shape[1]):
           print " %10.6f" % M[row,col],
       print
    return

def printArray(M, Ncol=None):
    if Ncol == None:
        Ncol = M.shape[0]

    for col in range(Ncol):
        print " %10.6f" % M[col],
    print
    return

def printArrayString(M):
    s = ''
    for col in range(M.shape[0]):
        s += " %10.6f" % M[col]
    s += '\n'
    return s

def printMatString(M):
    s = ''
    for row in range(M.shape[0]):
       for col in range(M.shape[1]):
           s += " %10.6f" % M[row,col]
       s += '\n'
    return s

def printGeomGrad(geom, grad):
    print "\tGeometry and Gradient"
    Natom = geom.shape[0]

    for i in range(Natom):
        print "\t%20.10f%20.10f%20.10f" % (geom[i,0], geom[i,1], geom[i,2])
    for i in range(Natom):
        print "\t%20.10f%20.10f%20.10f" % (grad[3*i+0], grad[3*i+1], grad[3*i+2])

### Simple chemical info.  Move elsewhere?

# return period from atomic number
def ZtoPeriod(Z):
    if   Z <=  2: return 1
    elif Z <= 10: return 2
    elif Z <= 18: return 3
    elif Z <= 36: return 4
    else:         return 5

# "Average" bond length given two periods
# Values below are from Lindh et al.
# Based on DZP RHF computations, I suggest: 1.38 1.9 2.53, and 1.9 2.87 3.40
def AverageRFromPeriods(perA, perB):
    if perA == 1:
        if   perB == 1: return 1.35
        elif perB == 2: return 2.1
        else:           return 2.53
    elif perA == 2:
        if   perB == 1: return 2.1
        elif perB == 2: return 2.87
        else:           return 3.40
    else:
        if   perB == 1: return 2.53
        else:           return 3.40

