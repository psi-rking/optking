#from math import fabs
#import numpy as np

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

